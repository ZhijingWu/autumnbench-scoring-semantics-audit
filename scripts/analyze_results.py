import pathlib,json,hashlib,collections,math
O=pathlib.Path(__file__).resolve().parents[1]
E=O/'execution_evidence';rows=[];stages={};replay=[]
def canonical(x):return json.dumps(x,sort_keys=True)
def paper(t,g):return 0 if t<g-1 else 1 if t in (g-1,g) else 1.377/(1-t/g*math.exp(-t/g))-1.178
for stage in ['primary','extension','support']:
 x=json.loads((E/f'results/{stage}_results.json').read_text());assert x['status']=='completed' and not x['failures']
 stages[stage]=x;rows.extend(x['observed_results']);replay.extend(x.get('paired_replays',[]))
traces=[]
for stage in stages:
 path=E/f'raw/actions_{stage}.jsonl'
 lines=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
 assert not any(x['event']=='action_exception' for x in lines)
 requests=[x for x in lines if x['event']=='action_request'];returns=[x for x in lines if x['event']=='action_return']
 assert len(requests)==len(returns)
 traces.append({'stage':stage,'jsonl_records':len(lines),'requests':len(requests),'returns':len(returns),'exceptions':0})
native=collections.Counter()
for stage in stages:
 counts=collections.Counter()
 for l in (E/f'raw/native_calls_{stage}.jsonl').read_text().splitlines():
  z=json.loads(l);assert z['error'] is None
  counts[z['scenario']+'/'+z['method']]+=1;native[z['method']]+=1
 assert dict(counts)==stages[stage]['native_call_counts']
 assert sum(counts.values())==stages[stage]['total_metered_native_calls']
for r in rows:
 assert r['status']=='COMPLETE'
 x=stages['primary' if r['environment']=='27VWC' else 'extension']
 q=next(y for y in x['paired_replays'] if y['seed']==r['seed'] and y['rep']==r['rep'])
 assert r['world_grids']==q['changed_grids'][:len(r['world_grids'])]
 assert r['official_scorer_inputs']['curr_frame']==len(r['prefix'])+r['browses']+1
 assert r['records'][-1]['terminal'] and r['records'][-1]['info']['terminal_condition']=='finish'
 assert all(v['observation']['grid']==r['world_grids'][r['selected_frame']] for v in r['records'] if v['kind']=='select')
 assert r['counterfactual_added_native_calls']==0 and r['official_state_unchanged_by_counterfactual']
 if r['selected_frame'] is not None:r['independent_observation_target_score']=paper(r['selected_frame'],q['first_observation_grid_divergence'])
 else:r['independent_observation_target_score']=None
 assert stages['primary']['controls']['meter_transparency'] and stages['primary']['controls']['direct_vs_composite_changed_returns']
groups=collections.defaultdict(list)
for r in rows:groups[(r['environment'],r['seed'],r['rep'],r['browses'],r['rule'].startswith('observed'))].append(r)
answer_groups=[v for v in groups.values() if len(v)==4]
assert len(answer_groups)==32
assert all(len(set(r['official_reward'] for r in v))==1 for v in answer_groups)
repeat_groups=collections.defaultdict(list)
for r in rows:repeat_groups[(r['environment'],r['seed'],r['rule'],r['browses'])].append(r)
assert all(len(v)==2 and canonical(v[0]['records'])==canonical(v[1]['records']) for v in repeat_groups.values())
browse_groups=collections.defaultdict(list)
for r in rows:
 if r['selected_frame'] is not None:browse_groups[(r['environment'],r['seed'],r['rep'],r['rule'])].append(r)
assert len(browse_groups)==48
assert all(min(v,key=lambda r:r['browses'])['official_reward']>max(v,key=lambda r:r['browses'])['official_reward'] for v in browse_groups.values())
assert all(len(set(r['counterfactual_index_only_reward'] for r in v))==1 for v in browse_groups.values())
summary={'factorial_scenarios':len(rows),'primary_factorial':52,'extension_factorial':52,'exploratory_support_factorial':32,
 'separate_transparency_path_controls':3,'total_cd_submissions':139,'paired_original_changed_replays':len(replay),'paired_world_steps_per_replay':128,
 'selected_scenarios':sum(r['selected_frame'] is not None for r in rows),
 'selected_score_changed_in_index_only_counterfactual':sum(r['selected_frame'] is not None and r['official_reward']!=r['counterfactual_index_only_reward'] for r in rows),
 'no_selection_controls':sum(r['selected_frame'] is None for r in rows),'same_cost_four_answer_groups':len(answer_groups),
 'same_cost_four_answer_groups_with_identical_official_scores':len(answer_groups),
 'browse_1_vs_4_groups':len(browse_groups),'browse_groups_with_official_decrease':len(browse_groups),
 'browse_groups_with_invariant_index_only_score':len(browse_groups),
 'exact_repeat_record_pairs':len(repeat_groups),'official_prefix_matches_paired_changed_replay':len(rows),
 'all_selected_returns_match_requested_frame':True,'all_terminal_finish':True,'all_jsonl_valid':True,
 'action_trace_counts':traces,'native_call_total':sum(native.values()),'native_call_method_counts':dict(native),
 'runtime_controls':{k:v for k,v in stages['primary']['controls'].items() if isinstance(v,bool)},
 'targets':[{ 'environment':env,'guard':stages[stage]['paired_replays'][0]['guard_time'],
  'first_grid_divergence':stages[stage]['paired_replays'][0]['first_observation_grid_divergence'],
  'replay_count':len(stages[stage]['paired_replays']),
  'original_all_seeds_and_repeats_same':len(set(q['original_trajectory_digest'] for q in stages[stage]['paired_replays']))==1,
  'changed_all_seeds_and_repeats_same':len(set(q['changed_trajectory_digest'] for q in stages[stage]['paired_replays']))==1,
  'random_tokens_present':stages[stage]['paired_replays'][0]['programs_contain_random_token']} for stage,env in [('primary','27VWC'),('extension','7WWW9')]]}
policies=[]
for seed in [0,1729]:
 for rep in [0,1]:
  subset=stages['support']['observed_results']
  a=next(x for x in subset if (x['seed'],x['rep'],x['rule'],x['browses'])==(seed,rep,'observed',4))
  b=next(x for x in subset if (x['seed'],x['rep'],x['rule'],x['browses'])==(seed,rep,'observed-2',1))
  assert a['world_grids']==b['world_grids']
  policies.append({'seed':seed,'rep':rep,'C_selected_frame':6,'C_browses':4,'E_selected_frame':4,'E_browses':1,
   'C_official_score':a['official_reward'],'E_official_score':b['official_reward'],
   'C_index_only_guard_score':a['counterfactual_index_only_reward'],'E_index_only_guard_score':b['counterfactual_index_only_reward'],
   'C_independent_observation_score':a['independent_observation_target_score'],'E_independent_observation_score':b['independent_observation_target_score'],
   'official_E_greater_C':b['official_reward']>a['official_reward'],
   'independent_C_greater_E':a['independent_observation_target_score']>b['independent_observation_target_score']})
summary['support_scripted_policy_comparison']=policies
summary['independence_warning']='Only2envs and1fixedworldprefix/env. Seeds/repetitions share deterministic outcomes; no population CI or learned-agent ranking claim.'
with (O/'ANALYSIS_SUMMARY.json').open('x') as f:json.dump(summary,f,indent=2)
flat=[]
for stage,x in stages.items():
 for r in x['observed_results']:
  flat.append({k:r.get(k) for k in ['scenario','environment','seed','rep','rule','browses','guard_time','selected_frame','official_reward','counterfactual_index_only_reward','independent_paper_formula_public_guard_proxy','independent_observation_target_score']}|
   {'official_scored_index':r['official_scorer_inputs']['curr_frame'],'world_steps':len(r['prefix']),'submission_feedback':r['records'][-1]['observation']['text'],'stage':stage})
with (O/'SCENARIO_RESULTS.json').open('x') as f:json.dump(flat,f,indent=2)
print(json.dumps(summary,indent=2))
