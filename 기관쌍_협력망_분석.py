from pathlib import Path
import argparse
import hashlib
import json
from itertools import combinations
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

CUTOFF = pd.Timestamp('2025-08-04')
CAPS = ['cap_infra_compute', 'cap_data', 'cap_model_pretraining',
        'cap_posttraining_alignment', 'cap_reasoning_safety_eval',
        'cap_deployment_optimization', 'cap_application_service']

def save(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')

def load_validate(data):
    names = ['기관_목록.csv', '검증된_기관쌍_협력관계.csv', '협력관계_출처.csv']
    n, r, s = [pd.read_csv(data / f, encoding='utf-8-sig') for f in names]
    required = [(n, ['node_id','label','actor_type','teams','is_lead','multi_team']),
                (r, ['relation_id','source','target','date','period','relation_type','evidence_level','evidence_summary','source_id']),
                (s, ['source_id','title','publisher','date','url'])]
    for df, cols in required:
        if not set(cols) <= set(df.columns):
            raise ValueError(f'Missing columns: {set(cols)-set(df.columns)}')
        if df[cols].isna().any().any():
            raise ValueError('Missing required values')
    for df, col in [(n,'node_id'), (r,'relation_id'), (s,'source_id')]:
        if df[col].duplicated().any():
            raise ValueError(f'Duplicate ID: {col}')
    if r.duplicated(subset=[c for c in r if c != 'relation_id']).any():
        raise ValueError('Duplicate relation content; review before counting weights')
    if not set(r.source) | set(r.target) <= set(n.node_id):
        raise ValueError('Unknown endpoint')
    if not set(r.source_id) <= set(s.source_id):
        raise ValueError('Unknown source reference')
    if (r.source == r.target).any():
        raise ValueError('Self-loop')
    if not r.evidence_level.isin([2,3]).all():
        raise ValueError('Evidence must be 2 or 3')
    for col in ['is_lead','multi_team']:
        if not n[col].isin([0,1]).all():
            raise ValueError(f'Invalid flag: {col}')
    n['team_set'] = n.teams.map(lambda x: set(x.split(';')))
    if not ((n.team_set.map(len) > 1).astype(int) == n.multi_team).all():
        raise ValueError('multi_team inconsistent with teams')
    r['date'] = pd.to_datetime(r.date, format='%Y-%m-%d', errors='raise')
    s['date'] = pd.to_datetime(s.date, format='%Y-%m-%d', errors='raise')
    derived = np.where(r.date < CUTOFF, 'pre', 'during')
    if not (derived == r.period).all():
        raise ValueError('Period inconsistent with cutoff (cutoff day is during)')
    merged = r.merge(s[['source_id','date']], on='source_id', suffixes=('', '_source'))
    checks = {'nodes':len(n), 'records':len(r), 'sources':len(s),
              'source_date_mismatch_count':int((merged.date != merged.date_source).sum()),
              'unused_sources':sorted(set(s.source_id)-set(r.source_id)),
              'shared_url_source_groups':s.groupby('url').source_id.apply(list).loc[lambda x:x.map(len)>1].tolist(),
              'replacement_character_count':sum(int(df.astype(str).apply(lambda x:x.str.count('\ufffd')).sum().sum()) for df in [n,r,s]),
              'sha256':{f:hashlib.sha256((data/f).read_bytes()).hexdigest() for f in names}}
    return n, r, s, checks

def graph(n, records):
    g = nx.Graph()
    g.add_nodes_from(n.node_id)
    for row in records.itertuples():
        if g.has_edge(row.source,row.target):
            g[row.source][row.target]['weight'] += 1
        else:
            g.add_edge(row.source,row.target,weight=1)
    nx.set_edge_attributes(g, {(u,v):1/d['weight'] for u,v,d in g.edges(data=True)}, 'distance')
    assert sum(dict(g.degree(weight='weight')).values()) == 2*len(records)
    assert set(map(frozenset,g.edges)) == {frozenset((a,b)) for a,b in zip(records.source, records.target)}
    return g

def metrics(g, n, period, analysis='baseline'):
    b = nx.betweenness_centrality(g, weight='distance', normalized=True)
    bu = nx.betweenness_centrality(g, weight=None, normalized=True)
    c = nx.closeness_centrality(g, distance='distance', wf_improved=True)
    cu = nx.closeness_centrality(g, wf_improved=True)
    cl = nx.clustering(g)
    cw = nx.clustering(g, weight='weight')
    comp = {v:i for i,vs in enumerate(sorted(nx.connected_components(g),key=lambda x:(-len(x),sorted(x)))) for v in vs}
    return [dict(analysis=analysis,period=period,node_id=row.node_id,label=row.label,
                 teams=row.teams,is_lead=row.is_lead,multi_team=row.multi_team,
                 degree=g.degree(row.node_id),degree_centrality=nx.degree_centrality(g)[row.node_id],
                 weighted_degree=g.degree(row.node_id,weight='weight'),
                 betweenness=b[row.node_id],betweenness_unweighted=bu[row.node_id],
                 closeness=c[row.node_id],closeness_unweighted=cu[row.node_id],
                 clustering=cl[row.node_id],clustering_weighted=cw[row.node_id],component_id=comp[row.node_id]) for row in n.itertuples()]

def summary(g, leads, period, analysis='baseline', **extra):
    horizontal = [(u,v) for u,v in g.edges if u not in leads and v not in leads]
    m = g.number_of_edges()
    strength = g.size(weight='weight')
    return dict(analysis=analysis, period=period,nodes=len(g),active_nodes=sum(d>0 for _,d in g.degree()),
                edges=m,relation_records=int(strength),density=nx.density(g),
                components=nx.number_connected_components(g),isolates=len(list(nx.isolates(g))),
                largest_component=max(map(len,nx.connected_components(g)),default=0),
                mean_degree=np.mean([d for _,d in g.degree()]) if len(g) else 0,
                mean_weighted_degree=2*strength/len(g) if len(g) else 0,
                average_clustering=nx.average_clustering(g) if len(g) else 0,
                average_clustering_weighted=nx.average_clustering(g,weight='weight') if m else 0,
                transitivity=nx.transitivity(g),horizontal_edges=len(horizontal),
                horizontal_edge_ratio=len(horizontal)/m if m else np.nan,
                horizontal_record_ratio=sum(g[u][v]['weight'] for u,v in horizontal)/strength if strength else np.nan,**extra)

def transitions(graphs, records):
    rows=[]
    for u,v in sorted(tuple(sorted(e)) for e in graphs['all'].edges):
        a=graphs['pre'].get_edge_data(u,v,{}).get('weight',0)
        b=graphs['during'].get_edge_data(u,v,{}).get('weight',0)
        rr=records[((records.source==u)&(records.target==v))|((records.source==v)&(records.target==u))]
        rows.append(dict(source=u,target=v,pre_weight=a,during_weight=b,all_weight=a+b,
                         transition='persisted' if a and b else 'pre_only' if a else 'newly_observed_during',
                         relation_ids=';'.join(rr.relation_id),source_ids=';'.join(sorted(set(rr.source_id)))))
    return rows

def brokers(g,n,period,analysis):
    attrs=n.set_index('node_id').to_dict('index')
    constraint=nx.constraint(g,weight='weight')
    between=nx.betweenness_centrality(g,weight='distance')
    rows=[]
    for row in n[n.multi_team==1].itertuples():
        v=row.node_id
        pairs=list(combinations(g.neighbors(v),2))
        open_pairs=[(a,b) for a,b in pairs if not g.has_edge(a,b)]
        cross=[(a,b) for a,b in open_pairs if attrs[a]['team_set']-{'EXTERNAL'} and attrs[b]['team_set']-{'EXTERNAL'} and not attrs[a]['team_set']&attrs[b]['team_set']]
        h=g.copy(); h.remove_node(v)
        # Fragmentation relative to the original component after removing the broker.
        own=nx.node_connected_component(g,v)-{v}
        fragments=nx.number_connected_components(h.subgraph(own)) if own else 0
        rows.append(dict(analysis=analysis,period=period,node_id=v,degree=g.degree(v),
                         weighted_degree=g.degree(v,weight='weight'),betweenness=between[v],
                         constraint=constraint[v],open_neighbor_pairs=len(open_pairs),
                         cross_team_open_neighbor_pairs=len(cross),
                         neighbor_team_count=len(set().union(*(attrs[x]['team_set']-{'EXTERNAL'} for x in g.neighbors(v)))),
                         articulation_point=v in set(nx.articulation_points(g)),
                         original_component_fragments_after_removal=fragments))
    return rows

def capability_analysis(profile,n,g):
    """Explicit opt-in only. Reject incomplete, negative, nonfinite or zero vectors.

    Scores must share a documented common scale. Tied maxima are all retained.
    Cross-capability means disjoint dominant-capability sets. Team coverage is
    the proportion of members scoring >0 in each capability, plus its mean.
    """
    if profile.node_id.duplicated().any() or set(profile.node_id)!=set(n.node_id):
        raise ValueError('Capability profile must contain exactly the master node IDs')
    x=profile.set_index('node_id')[CAPS].apply(pd.to_numeric,errors='raise').reindex(n.node_id)
    a=x.to_numpy(float)
    if not np.isfinite(a).all() or (a<0).any() or (np.linalg.norm(a,axis=1)==0).any():
        raise ValueError('Fill every capability with finite nonnegative scores and nonzero vectors first')
    norm=a/np.linalg.norm(a,axis=1,keepdims=True)
    sim=pd.DataFrame(norm@norm.T,index=x.index,columns=x.index)
    dominant={v:set(x.columns[x.loc[v]==x.loc[v].max()]) for v in x.index}
    edges=[dict(source=u,target=v,weight=d['weight'],cosine_similarity=sim.loc[u,v],
                dominant_source=';'.join(sorted(dominant[u])),dominant_target=';'.join(sorted(dominant[v])),
                cross_capability=not bool(dominant[u]&dominant[v])) for u,v,d in g.edges(data=True)]
    coverage=[]
    for team in sorted(set().union(*n.team_set)-{'EXTERNAL'}):
        ids=n.loc[n.team_set.map(lambda t:team in t),'node_id']
        for cap in CAPS:
            coverage.append(dict(team=team,capability=cap,member_count=len(ids),
                                 positive_member_ratio=(x.loc[ids,cap]>0).mean(),mean_score=x.loc[ids,cap].mean()))
    return sim, pd.DataFrame(edges), pd.DataFrame(coverage)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent/'data')
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'output')
    parser.add_argument('--capability-profile',type=Path,default=None)
    args=parser.parse_args(); out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    n,r,s,validation=load_validate(args.data_dir)
    leads=set(n.loc[n.is_lead==1,'node_id']); teams=sorted(set().union(*n.team_set)-{'EXTERNAL'})
    graphs={p:graph(n,r if p=='all' else r[r.period==p]) for p in ['pre','during','all']}
    summaries=[]; node_rows=[]; team_rows=[]; broker_rows=[]; strict_rows=[]; type_rows=[]
    for analysis,records in [('baseline',r),('strict',r[r.evidence_level==3])]:
        for p in graphs:
            rr=records if p=='all' else records[records.period==p]
            g=graph(n,rr); sm=summary(g,leads,p,analysis)
            summaries.append(sm); node_rows.extend(metrics(g,n,p,analysis)); broker_rows.extend(brokers(g,n,p,analysis))
            if analysis=='strict':
                base=next(x for x in summaries if x['analysis']=='baseline' and x['period']==p)
                strict_rows.append(dict(**sm,edge_retention=g.number_of_edges()/base['edges'] if base['edges'] else np.nan,
                                        edge_change=g.number_of_edges()-base['edges'],record_change=len(rr)-base['relation_records']))
            for team in teams:
                ids=set(n.loc[n.team_set.map(lambda t:team in t),'node_id'])
                h=g.subgraph(ids)
                boundary=sum((u in ids)!=(v in ids) for u,v in g.edges)
                team_rows.append(summary(h,leads,p,analysis,team=team,boundary_edges=boundary))
            for typ in sorted(r.relation_type.unique()):
                tr=rr[rr.relation_type==typ]; tg=graph(n,tr)
                type_rows.append(summary(tg,leads,p,analysis,relation_type=typ,unique_sources=tr.source_id.nunique()))
    transition=transitions(graphs,r)
    save(node_rows,out/'기관별_네트워크_지표.csv'); save(summaries,out/'네트워크_요약.csv')
    save(transition,out/'협력관계_기간별_변화.csv'); save(team_rows,out/'팀별_네트워크_요약.csv')
    save(broker_rows,out/'다중팀_기관_중개_분석.csv'); save(strict_rows,out/'강한_근거_민감도_분석.csv')
    save(type_rows,out/'협력유형별_네트워크_요약.csv')
    template=n[['node_id','label','teams']].copy()
    for cap in CAPS: template[cap]=np.nan
    # Never overwrite a profile that a researcher has already populated.
    if not (out/'기관별_역량_입력_양식.csv').exists():
        save(template,out/'기관별_역량_입력_양식.csv')
    if args.capability_profile:
        sim,edges,coverage=capability_analysis(pd.read_csv(args.capability_profile),n,graphs['all'])
        sim.to_csv(out/'capability_cosine_similarity.csv',encoding='utf-8-sig')
        save(edges,out/'capability_edge_linkage.csv'); save(coverage,out/'team_capability_coverage.csv')
    colors=plt.get_cmap('tab10'); team_color={t:colors(i) for i,t in enumerate(teams+['EXTERNAL'])}
    attrs=n.set_index('node_id').to_dict('index')
    pos={}
    components=sorted((c for c in nx.connected_components(graphs['all']) if len(c)>1),key=lambda c:(-len(c),sorted(c)))
    for i,component in enumerate(components):
        local=nx.kamada_kawai_layout(graphs['all'].subgraph(sorted(component)),weight='distance')
        scale=.95 if i==0 else .36
        center=np.array([-.3,.25]) if i==0 else np.array([1.25,.5-(i-1)*.85])
        pos.update({v:xy*scale+center for v,xy in local.items()})
    isolates=sorted(nx.isolates(graphs['all']))
    for i,v in enumerate(isolates):
        pos[v]=np.array([-1.3+(i%7)*.47,-1.15-(i//7)*.25])
    for p,g in graphs.items():
        fig,ax=plt.subplots(figsize=(19,14))
        for lead,shape in [(False,'o'),(True,'s')]:
            ids=[v for v in g if bool(attrs[v]['is_lead'])==lead]
            nx.draw_networkx_nodes(g,pos,nodelist=ids,node_shape=shape,
                node_size=[170+75*g.degree(v) for v in ids],
                node_color=[team_color[sorted(attrs[v]['team_set'])[0]] if len(attrs[v]['team_set'])==1 else '#bdbdbd' for v in ids],
                edgecolors='black',linewidths=.7,ax=ax)
        nx.draw_networkx_edges(g,pos,width=[.8+d['weight'] for _,_,d in g.edges(data=True)],alpha=.55,ax=ax)
        label_pos={v:xy+np.array([0,.043]) for v,xy in pos.items()}
        nx.draw_networkx_labels(g,label_pos,font_size=8,ax=ax)
        for t,color in team_color.items(): ax.scatter([],[],color=color,label='외부기관' if t=='EXTERNAL' else t)
        ax.scatter([],[],color='#bdbdbd',label='다중 팀 소속 기관')
        period_label={'pre':'선정 이전','during':'선정 이후','all':'전체 기간'}[p]
        ax.set_title(f'검증된 기관쌍 협력관계 | {period_label} | 기관 {len(g)}개, 협력관계 {g.number_of_edges()}개',fontsize=17)
        ax.text(.01,.01,'사각형: 주관기관 | 노드 크기: 연결 기관 수 | 선 두께: 관계 기록 수 | 모든 기간에 동일한 노드 배치 사용\n고립기관 포함. 팀 소속 정보만으로 협력관계를 생성하지 않음.',transform=ax.transAxes,fontsize=10)
        ax.legend(loc='upper left',frameon=False); ax.axis('off'); fig.tight_layout()
        fig.savefig(out/{'pre':'선정_이전_협력망.png','during':'선정_이후_협력망.png','all':'전체_기간_협력망.png'}[p],dpi=170); plt.close(fig)
    (out/'데이터_검증_결과.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
    report(n,r,s,validation,summaries,node_rows,transition,team_rows,broker_rows,strict_rows,type_rows,out)
    print(json.dumps({'validation':validation,'baseline_summary':[x for x in summaries if x['analysis']=='baseline'],
                      'transitions':pd.DataFrame(transition).transition.value_counts().to_dict()},ensure_ascii=True,indent=2))

def table(rows,cols):
    df=pd.DataFrame(rows)[cols]
    return df.to_markdown(index=False,floatfmt='.4f')

def report(n,r,s,v,summaries,nodes,trans,teams,brokers,strict,types,out):
    baseline=[x for x in summaries if x['analysis']=='baseline']
    lead=[x for x in nodes if x['analysis']=='baseline' and x['is_lead']]
    multi=[x for x in brokers if x['analysis']=='baseline']
    counts=pd.DataFrame(trans).transition.value_counts().to_dict()
    text=f'''# 검증된 기관쌍 협력관계 네트워크 분석

이 연구는 컨소시엄 명단이 아니라 검증된 기관쌍 협력관계를 분석한다. DARPA형 자율적 컨소시엄 구성 아래 독파모 참여기관의 관측 가능한 기술협력 구조를 기술한다. 입력 CSV의 relation record를 직접협업 근거로 받아들였으며 원문 웹자료를 재검증하거나 추가 자료를 수집하지 않았다.

## A. 데이터 검증과 분석 범위

기관 {len(n)}개, 관계 기록 {len(r)}개, 출처 ID {len(s)}개. 필수값·ID 유일성·관계 내용 중복·기관 및 출처 참조·자기연결 부재·근거수준·팀 플래그·기간 일치를 검증했다. 날짜와 출처 날짜 불일치 {v['source_date_mismatch_count']}건, Unicode 대체문자 {v['replacement_character_count']}개. 동일 URL을 공유하는 출처 ID 묶음: {v['shared_url_source_groups']}. 같은 URL 또는 같은 출처의 복수 기록은 독립적인 증거의 수를 의미하지 않는다. 입력 해시는 데이터_검증_결과.json에 보존했다.

pre는 2025-08-04 미만, during은 당일 이상이다. 기간은 관계 기록의 날짜를 분류한 관측 창이며 협력의 실제 시작·종료 또는 지속 여부를 확정하지 않는다. pre/during 관측 기간 길이는 동일하지 않으며 발생률로 비교하지 않는다. 외부기관도 master에 있으면 포함한다. 모든 전체 그래프는 고립기관을 포함한 동일 기관 집합을 사용한다. 팀 소속은 제공된 정적 메타데이터이며 pre 시점 소속의 역사적 복원이 아니다.

## B–C. 네트워크 구조

관측 edge는 pre 12개에서 during 35개로, 최대 연결요소는 4개 기관에서 31개 기관으로 달라진다. 반면 평균 clustering은 약 0.0881에서 0.0396으로 낮아진다. 이는 관측 연결의 확대와 삼각형 폐쇄 정도가 같은 방향으로 움직이지 않음을 보여준다. all에서는 42개 고유 기관쌍이 확인된다.

{table(baseline,['period','nodes','active_nodes','edges','relation_records','density','components','isolates','largest_component','average_clustering'])}

무방향 단순 그래프에서 기관쌍별 weight는 해당 기간 relation record 수다. all은 원시 기록의 합집합을 집계한다. 동일 소속이나 다자행사 참여만으로 edge를 생성하지 않았다. degree는 이웃 수, weighted degree는 기록 수 합이다. betweenness는 정규화하고 거리=1/weight를 사용한다. closeness는 같은 거리와 NetworkX의 WF 도달가능성 보정을 적용하므로 비연결성을 반영하지만 강한 edge는 값이 1을 넘게 할 수도 있다. 비교용 무가중 betweenness/closeness도 제공한다. clustering은 무가중 및 NetworkX의 최대 가중치 정규화 방식 모두 제공한다. density는 무가중 edge 수 / 가능한 기관쌍 수이며, 평균 clustering은 고립기관을 포함한다. components에는 고립기관도 포함한다. 가중치는 협력 품질·투입량·독립 출처 수를 뜻하지 않는다.

## D. 주관기관 중심성

{table(lead,['period','node_id','degree','weighted_degree','betweenness','closeness'])}

각 지표는 전체 53개 기관을 기준으로 산출하며, 주관기관의 기간별 기록상 위치를 비교한다. 높은 degree는 확인된 상대기관이 많음을 뜻하며 실제 영향력이나 사업 성과의 직접 측정은 아니다.

## E. 수평적 관계

{table(baseline,['period','horizontal_edges','edges','horizontal_edge_ratio','horizontal_record_ratio'])}

horizontal은 양 끝점 모두 master의 is_lead=0인 edge다. 분모는 외부기관을 포함한 전체 관측 edge 수이며, 기록가중 비율도 별도로 제시한다. 외부기관도 비주관기관으로 포함되므로 이 비율은 컨소시엄 내부 구성원 사이의 수평협력 비율과 동일하지 않다.

## F. 다중 팀 기관의 brokerage

baseline during에서 KAIST의 가중 betweenness는 약 0.1825, 고려대학교는 0.1425, SNU는 0.0196이다. strict during에서는 KAIST와 SNU가 고립되고 고려대학교는 약 0.0045로 낮아진다. 따라서 다중 팀 기관의 관측 중개 위치는 수준 2 역할협업 기록의 포함 여부에 민감하다. 다중 소속 자체가 강한 직접 공동개발 관계를 보장하지 않는다.

{table(multi,['period','node_id','degree','betweenness','constraint','open_neighbor_pairs','cross_team_open_neighbor_pairs','articulation_point'])}

multi_team 자체는 연결 근거가 아니다. open_neighbor_pairs는 해당 기관의 이웃 중 서로 직접 연결되지 않은 쌍 수다. cross_team은 두 이웃의 비외부 팀 집합이 모두 비어 있지 않고 서로 겹치지 않는 경우만 센다. 단순 팀 수를 중개로 간주하지 않는다. constraint는 기록가중 Burt constraint이며 고립기관은 결측이다. articulation은 제거 시 연결요소를 분리하는 위상적 위치로, 실제 지식흐름이나 인과적 매개를 입증하지 않는다. 제거 후 원래 소속 연결요소의 잔여 조각 수를 CSV에 제공한다.

## G. 기관쌍 transition

{counts}

persisted는 두 기간 모두 기록이 관측된 기관쌍, pre_only는 pre에만 기록된 기관쌍, newly_observed_during은 during에만 기록된 기관쌍이다. 후자는 **newly observed after selection**으로만 해석한다. persisted는 지속 활동의 입증이 아니고 pre_only는 협력 종료의 입증이 아니다. 새 관측 관계를 독파모 선정의 인과효과로 표현할 수 없다.

## H. 팀별 유도 부분그래프

{table([x for x in teams if x['analysis']=='baseline'],['period','team','nodes','edges','density','components','horizontal_edge_ratio','boundary_edges'])}

양 끝점이 해당 팀 명단에 있는 관측 edge만 포함한다. 다중 팀 기관은 각 팀에 중복 포함되어 팀별 결과를 합산하면 전체와 일치하지 않는다. 외부기관은 팀 내부 그래프에서 제외하며 boundary_edges는 팀 안팎을 잇는 관측 edge다. 팀 크기·고립기관 구성·공개 보도량이 달라 단순 순위는 성과 순위가 아니다.

## I. evidence_level=3 엄격 민감도 분석

strict during의 edge는 18개로 baseline 35개의 약 51.4%가 유지된다. 연결요소는 20개에서 36개로 늘어난다. pre의 12개 edge는 모두 수준 3으로 유지된다. 기간 차이의 일부는 수준 2 역할협업 기록에 의존하며, 강한 근거만으로도 during edge가 pre보다 많다는 관측과 함께 해석해야 한다.

{table(strict,['period','edges','relation_records','edge_retention','density','components','horizontal_edge_ratio','average_clustering'])}

수준 3은 직접 공동개발·공동연구·실증 명시, 수준 2는 구체적 역할협업 확인이다. 엄격 분석은 기록 단위로 수준 2를 제거한 뒤 기관쌍을 다시 집계한다. 같은 쌍의 수준 3 기록이 남으면 edge도 남는다. node_metrics, network_summary, team_summary, broker_analysis, relation_type_summary에는 baseline과 strict를 모두 제공한다. 수준 2 제거로 달라지는 연결성·주관기관 중심성·다중 팀 중개를 함께 검토해야 하며 명단에 있는 기관의 실제 비협력을 추정해서는 안 된다.

## J. relation_type별 네트워크·빈도

{table([x for x in types if x['analysis']=='baseline' and x['period']=='all'],['relation_type','relation_records','edges','active_nodes','unique_sources','density','components'])}

CSV에는 각 유형의 pre/during/all 및 strict 빈도와 부분네트워크 지표를 제공한다. 유형별로 기록을 필터링한 뒤 동일 기관쌍을 합친다. 하나의 기관쌍이 여러 유형에 속할 수 있어 유형별 edge 수를 전체 edge 수와 합산하지 않는다. 유형 명칭은 입력값 그대로 유지했다.

## K. 해석 한계

공개자료 selection bias 때문에 보도·공시가 많은 기관과 협업이 과대표집될 수 있다. 미관측 edge는 관계 부재가 아니다. 검증된 기록이라는 데이터 정의는 원문 자체의 완전성·정확성에 대한 독립 감사와 다르다. 동일 발표에서 추출한 기록들은 서로 의존하며 가중치는 공개 기록의 반복성도 반영한다. 선정 전후의 노출시간·공개 관행·기관별 홍보 차이를 통제하지 않았고 비교집단도 없으므로 DARPA형 구성이나 선정의 인과효과를 추정하지 않는다. 본 결과는 공급된 세 CSV에 한정된 기술적 네트워크 비교다.

## 향후 R 텍스트 역량 분석 결합

기관별_역량_입력_양식.csv는 기관 ID별 7개 역량 값이 모두 비어 있다. 임의 역량값, 유사도, 우세 역량 또는 역량 간 관계를 생성하지 않았다. 연구자가 공통 척도로 모든 값을 입력하고 명시적으로 --capability-profile을 전달했을 때만 함수가 계산된다. 결측·무한·음수·영벡터는 오류로 중단한다. cosine은 모든 기관쌍의 속성 유사도이며 새로운 협력 edge가 아니다. dominant는 공동 최댓값을 모두 유지한다. cross-capability linkage는 기존 edge의 두 끝점 우세역량 집합이 겹치지 않는 경우다. team coverage는 역량별 양수 점수 구성원 비율과 평균점수이며 팀 구성원 중복을 허용한다. 텍스트 코딩·척도 정규화·양수의 의미를 별도로 문서화한 후 사용해야 한다.

![pre](선정_이전_협력망.png)
![during](선정_이후_협력망.png)
![all](전체_기간_협력망.png)

세 그림은 all 그래프의 연결요소별 Kamada–Kawai 좌표를 공통으로 사용하고 all 고립기관은 하단에 배치한다. 연결요소별 화면 배율은 다르다. 선 두께는 기록 수, 노드 크기는 각 기간 degree, 사각형은 주관기관, 회색은 다중 팀 기관이다. 좌표는 관계의 관측 날짜나 실제 지리적 위치를 뜻하지 않는다.
'''
    (out/'기관쌍_협력망_분석_보고서.md').write_text(text,encoding='utf-8')

if __name__=='__main__':
    main()
