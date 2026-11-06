"""Unrelated neutral throughput check; no study prompts and no changes to the frozen runner."""
import sys, time
sys.path.insert(0, '/workspace/aster-next-turn/eval/painaxis/next_turn')
import collect as E
import common as C
import materials as M
C.check_implementation()
t0=time.monotonic()
model,tok=E.load()
load_seconds=time.monotonic()-t0
first=[{'role':'system','content':M.CONFIG['system']},{'role':'user','content':'Explain how the water cycle moves water between the ocean, atmosphere, and land. Use three short paragraphs suitable for a general audience.'}]
ids=E.render(tok,first)
initial,obs=E.sample(model,tok,ids,1903,256,E.boundary_index(tok,first))
eos=model.generation_config.eos_token_id
eos=eos if isinstance(eos,list) else [eos]
second,qids=E.continuation(tok,first,ids,initial['response_ids'],'Which term describes water vapor turning into liquid? Reply with exactly A or B. A: condensation. B: evaporation.',eos)
choice,obs2=E.sample(model,tok,qids,1904,8)
receipt={'created':C.now(),'purpose':'neutral throughput only; not part of the scientific sample','load_seconds':load_seconds,'initial_messages':first,'initial_prefill_tokens':len(ids),'initial':initial,'choice_messages':second,'choice_prefill_tokens':len(qids),'choice':choice,'source_sha256':C.filehash(__file__),'implementation_sha256':C.check_implementation()}
C.save_new(C.RUN/'NEUTRAL_BENCHMARK.json',receipt)
print({'initial_tokens':len(initial['response_ids']),'initial_seconds':initial['seconds'],'choice_tokens':len(choice['response_ids']),'choice_seconds':choice['seconds'],'load_seconds':load_seconds},flush=True)
