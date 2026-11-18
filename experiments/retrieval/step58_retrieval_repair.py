"""Step 58 exploratory QASPER retrieval-repair comparisons (document scoped).

Run from TrustRAG_organized:
 python experiments/retrieval/step58_retrieval_repair.py --input data/qasper/pilot_v1 --limit 5
 python experiments/retrieval/step58_retrieval_repair.py --input data/qasper/pilot_v1

Gold is never loaded by retrieve(); evaluation occurs after prediction files are saved.
Methods: dense30 rerank, dense50 rerank, BM25+Dense RRF30 rerank, dense30 rank fusion.
All models and methods use a fixed original question (no LLM query rewriting in this step).
"""
import argparse, csv, hashlib, json, math, re, time
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

METHODS=['dense30_rerank','dense50_rerank','hybrid30_rerank','dense30_fusion','dense30_raw']

def read_json(p): return json.loads(p.read_text(encoding='utf-8'))
def read_jsonl(p):
    with p.open(encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]
def save_csv(path, rows, columns):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def tokens(s): return re.findall(r"(?u)\b\w+\b",str(s).lower())

class BM25:
    def __init__(self, texts, k1=1.5, b=.75):
        self.docs=[Counter(tokens(t)) for t in texts]; self.lengths=[sum(d.values()) for d in self.docs]
        self.avg=sum(self.lengths)/max(len(self.docs),1);self.n=len(self.docs);self.k1=k1;self.b=b
        df=Counter();[df.update(d.keys()) for d in self.docs]
        self.idf={w:math.log(1+(self.n-f+.5)/(f+.5)) for w,f in df.items()}
    def score(self,q):
        terms=set(tokens(q));scores=np.zeros(self.n)
        for i,d in enumerate(self.docs):
            norm=self.k1*(1-self.b+self.b*self.lengths[i]/max(self.avg,1e-9))
            scores[i]=sum(self.idf.get(w,0)*d[w]*(self.k1+1)/(d[w]+norm) for w in terms if d[w])
        return scores

def rrf(indices_a,indices_b,n,constant=60):
    score=np.zeros(n)
    for order in (indices_a,indices_b):
        for rank,i in enumerate(order,1):score[int(i)]+=1/(constant+rank)
    return np.argsort(-score,kind='stable')

def retrieve(root,questions,units,out,model_id,reranker_id):
    from sentence_transformers import SentenceTransformer,CrossEncoder
    bydoc=defaultdict(list)
    ids=set()
    for u in units:
        uid=str(u['paragraph_id'])
        if uid in ids:raise ValueError('Duplicate unit ID: '+uid)
        ids.add(uid)
        if str(u.get('text','')).strip():bydoc[str(u['document_id'])].append(u)
    for q in questions:
        if str(q['document_id']) not in bydoc:raise ValueError('Missing doc '+str(q['document_id']))
    print('Loading dense model:',model_id,flush=True)
    embed=SentenceTransformer(model_id)
    vectors={};bm25={}
    for did,doc in bydoc.items():
        vectors[did]=embed.encode([u['text'] for u in doc],batch_size=32,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False).astype('float32')
        bm25[did]=BM25([u['text'] for u in doc])
    print('Loading reranker:',reranker_id,flush=True)
    cross=CrossEncoder(reranker_id)
    predictions=[];cost=[]
    for qi,q in enumerate(questions,1):
        t0=time.perf_counter();did=str(q['document_id']);qid=str(q['question_id']);question=str(q['question']);doc=bydoc[did]
        qvec=embed.encode([question],normalize_embeddings=True,convert_to_numpy=True).astype('float32')[0]
        dense_scores=vectors[did]@qvec
        dense_order=np.argsort(-dense_scores,kind='stable')
        bm_scores=bm25[did].score(question);bm_order=np.argsort(-bm_scores,kind='stable')
        hybrid_order=rrf(dense_order,bm_order,len(doc))
        pools={'dense30_rerank':dense_order[:30], 'dense50_rerank':dense_order[:50],
               'hybrid30_rerank':hybrid_order[:30]}
        rescored={};total_pairs=0
        for pool_name,order in pools.items():
            # Cross-encoder score pairs shared between arms, but count each arm's logical workload separately.
            scores=np.asarray(cross.predict([(question,doc[int(i)]['text']) for i in order],show_progress_bar=False)).reshape(-1)
            rescored[pool_name]={int(i):float(s) for i,s in zip(order,scores)}
            total_pairs+=len(order)
        result={}
        for name,order in pools.items():
            scores=rescored[name]
            result[name]=sorted((int(i) for i in order),key=lambda i:(-scores[i],i))[:10]
        top30=pools['dense30_rerank'];ce=rescored['dense30_rerank']
        cross_order=sorted((int(i) for i in top30),key=lambda i:(-ce[i],i))
        # Equal-weight reciprocal-rank fusion: dense top30 rank and cross-encoder top30 rank.
        fused_scores=defaultdict(float)
        for ordering in (top30,cross_order):
            for j,i in enumerate(ordering,1):fused_scores[int(i)]+=1/(60+j)
        result['dense30_fusion']=sorted((int(i) for i in top30),key=lambda i:(-fused_scores[i],i))[:10]
        result['dense30_raw']=[int(i) for i in dense_order[:10]]
        for name,positions in result.items():
            for rank,index in enumerate(positions,1):
                u=doc[index];predictions.append(dict(question_id=qid,document_id=did,method=name,rank=rank,
                                     unit_id=str(u['paragraph_id']),record_type=u.get('record_type',''),
                                     dense_score=round(float(dense_scores[index]),8)))
        cost.append(dict(question_id=qid,document_id=did,doc_units=len(doc),
                         dense_embeddings_query=1,bm25_query=1,
                         rerank_pairs_dense30=len(pools['dense30_rerank']),
                         rerank_pairs_dense50=len(pools['dense50_rerank']),
                         rerank_pairs_hybrid30=len(pools['hybrid30_rerank']),
                         wall_seconds=round(time.perf_counter()-t0,4)))
        if qi%10==0 or qi==len(questions):print(f'Retrieved {qi}/{len(questions)}',flush=True)
    save_csv(out/'predictions.csv',predictions,['question_id','document_id','method','rank','unit_id','record_type','dense_score'])
    save_csv(out/'cost_by_question.csv',cost,list(cost[0]) if cost else ['question_id'])
    return predictions,cost

def evaluate(root,qs,pred,out):
    # Gold NEVER influences retrieval, scores, or method rankings. Only offline scoring.
    alignment=root/'evaluation_v2'/'gold_structural_alignment.csv'
    gold=read_json(root/'gold_references_DO_NOT_USE_FOR_ROUTING.json')
    with alignment.open(encoding='utf-8',newline='') as f: align=list(csv.DictReader(f))
    ann_info=defaultdict(lambda:defaultdict(lambda:{'ids':set(),'bad':False,'types':set()}))
    for r in align:
        qid=str(r['question_id']);aid=str(r['annotation_id']);g=ann_info[qid][aid]
        ids=[x for x in str(r.get('matched_paragraph_ids','')).split('|') if x]
        if r['status'] not in ('exact','contained') or len(ids)!=1:g['bad']=True
        else:g['ids'].add(ids[0]);g['types'].add(r.get('record_type',''))
    ranked=defaultdict(list)
    for row in pred:ranked[(row['question_id'],row['method'])].append((row['rank'],row['unit_id']))
    for key in ranked:ranked[key]=[u for _,u in sorted(ranked[key])]
    refs={str(q['question_id']):q for q in gold};output=[]
    for q in qs:
        qid=str(q['question_id']);reference=refs[qid];ann_sets=[];float_proxy=False
        for k,a in enumerate(reference['annotations']):
            if a.get('unanswerable'):continue
            aid=str(a.get('annotation_id') or 'index-'+str(k)); info=ann_info[qid][aid]
            if a.get('evidence') and info['ids'] and not info['bad']:
                ann_sets.append(info['ids']);float_proxy|=('float_caption' in info['types'])
        union=set().union(*ann_sets) if ann_sets else set()
        status='evaluable_with_float_proxy' if union and float_proxy else ('evaluable' if union else ('no_aligned_gold' if any(not a.get('unanswerable') for a in reference['annotations']) else 'unanswerable'))
        for method in METHODS:
            ids=ranked[(qid,method)];row=dict(question_id=qid,document_id=str(q['document_id']),method=method,status=status,
                     n_gold_union=len(union),n_gold_annotations=len(ann_sets),has_float_caption=int(float_proxy))
            for k in [5,10]:
                found=set(ids[:k])&union
                row[f'hit@{k}']=int(bool(found)) if union else ''
                row[f'recall@{k}']=round(len(found)/len(union),6) if union else ''
                row[f'annotator_recall@{k}']=round(float(np.mean([len(set(ids[:k])&s)/len(s) for s in ann_sets])),6) if ann_sets else ''
            first=next((i for i,u in enumerate(ids[:10],1) if u in union),None)
            row['mrr@10']=round(1/first,6) if first else (0. if union else '')
            output.append(row)
    save_csv(out/'question_metrics.csv',output,list(output[0]))
    summary=[]
    for method in METHODS:
        for scope in ['all_aligned','non_float_only']:
            part=[r for r in output if r['method']==method and (str(r['status']).startswith('evaluable') if scope=='all_aligned' else r['status']=='evaluable')]
            record=dict(method=method,scope=scope,n_questions=len(part))
            for metric in ['hit@5','recall@5','annotator_recall@5','hit@10','recall@10','annotator_recall@10','mrr@10']:
                record[metric]=round(float(np.mean([r[metric] for r in part])),6) if part else ''
            summary.append(record)
    save_csv(out/'summary.csv',summary,list(summary[0]))
    return summary,output

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('data/qasper/pilot_v1'))
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--out',type=Path)
    p.add_argument('--embedding-model',default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--reranker-model',default='cross-encoder/ms-marco-MiniLM-L6-v2')
    args=p.parse_args()
    if args.limit<0:p.error('--limit must be >=0')
    root=args.input;out=args.out or root/'retrieval_step58';out.mkdir(parents=True,exist_ok=True)
    up=root/'runtime_adapter_v2'/'evidence_units_runtime.jsonl';qp=root/'questions_runtime.json'
    units=read_jsonl(up);qs=read_json(qp)
    if args.limit:qs=qs[:args.limit]
    pred,cost=retrieve(root,qs,units,out,args.embedding_model,args.reranker_model)
    summary,details=evaluate(root,qs,pred,out)
    manifest=dict(stage='exploratory',n_questions=len(qs),methods=METHODS,
       embedding_model=args.embedding_model,reranker_model=args.reranker_model,
       hashes={'units':digest(up),'questions':digest(qp),'gold_alignment':digest(root/'evaluation_v2'/'gold_structural_alignment.csv')},
       fixed_parameters=dict(dense_pool_base=30,dense_pool_expanded=50,hybrid_pool=30,rrf_constant=60,top_k=10,bm25_k1=1.5,bm25_b=.75),
       cost_notes='Rerank pair counts per arm are logical costs, wall time combines all methods and cannot represent individual arm latency.',
       cautions=['Fixed pilot is exploratory, not independent test.','Float captions are proxy evidence.','Original questions only, no query reformulation arm.','No gold available to runtime retrieval.'])
    (out/'protocol.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    for row in summary:print(row)
    print('Saved:',out)
if __name__=='__main__':main()
