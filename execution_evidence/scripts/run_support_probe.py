"""Real pinned interpreter/environment probes; no model calls or original writes."""
import os,sys,pathlib,json,time,hashlib,copy,math,traceback,collections,socket
ROOT=pathlib.Path('external/local-resource')
OLD=pathlib.Path('external/local-resource')
os.chdir(ROOT)
sys.dont_write_bytecode=True
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                  MPLBACKEND='Agg',MPLCONFIGDIR=str(ROOT/'scratch/mpl'),PYTHONDONTWRITEBYTECODE='1')
def denied(*a,**k):raise RuntimeError('Network denied in scientific probe')
socket.create_connection=denied
socket.socket.connect=denied
for p in [OLD/'sources/MARAProtocol_source',OLD/'build/native',
          pathlib.Path('external/local-resource'),
          pathlib.Path('external/local-resource'),
          OLD/'deps',ROOT/'adapters/generated/mara',ROOT/'adapters']:
    sys.path.insert(0,str(p))
import interpreter_module as native
sys.modules['python_examples.autumnbench.interpreter_module']=native
from generated.mara import mara_environment_pb2 as pb
sys.modules['mara_environment_pb2']=pb
from generated.mara import mara_environment_service_pb2 as spb
sys.modules['mara_environment_service_pb2']=spb
from generated.mara import mara_environment_service_pb2_grpc as sgrpc
sys.modules['mara_environment_service_pb2_grpc']=sgrpc
from python_examples.autumnbench import concrete_envs as official,environment_interfaces as interfaces,env_utils
from python_examples.autumnbench.autumnstdlib import autumnstdlib
import yaml

DATA=OLD/'sources/autumnbench_public'
P=json.loads((ROOT/'manifests/PREREGISTRATION.json').read_text())
STAGE=sys.argv[1]
assert STAGE=='support'
OUTFILE=ROOT/'results/support_results.json'
assert not OUTFILE.exists(),'Existing results preserved; use a new run ID'
CALLS=collections.Counter()
CALLLOG=(ROOT/'raw'/f'native_calls_{STAGE}.jsonl').open('x',buffering=1)
TRACE=(ROOT/'raw'/f'actions_{STAGE}.jsonl').open('x',buffering=1)
CURRENT='startup'

def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()
def emit(obj):TRACE.write(json.dumps(obj,default=str)+'\n');TRACE.flush()
class Meter:
    def __init__(self,label):self.inner=native.Interpreter();self.label=label
    def __getattr__(self,name):
        value=getattr(self.inner,name)
        if not callable(value):return value
        def call(*args,**kwargs):
            start=time.monotonic();result=None;error=None
            try:result=value(*args,**kwargs);return result
            except BaseException as e:error=repr(e);raise
            finally:
                CALLS[self.label+'/'+name]+=1
                CALLLOG.write(json.dumps({'scenario':self.label,'method':name,
                   'args_sha256':hashlib.sha256(repr((args,kwargs)).encode()).hexdigest(),
                   'result':result,'error':error,'elapsed_seconds':time.monotonic()-start},default=str)+'\n')
                if sum(CALLS.values())>P['resource_limits']['native_call_cap']:raise RuntimeError('Native call budget exceeded')
        return call

def state(ui):
    return {n:getattr(ui,n,None) for n in ['time','curr_frame','trigger_start_time','triggering_state','state','seed','inited','is_terminal']}|{'frame_count':len(getattr(ui,'frames',[]))}
def grid_from_obs(text):
    i=text.find('[[')
    if i<0:return None
    return json.JSONDecoder().raw_decode(text[i:])[0]
def obs_record(obs):
    return {'text':obs.text_data,'image_hex':obs.image_data.hex(),
            'serialized_hex':obs.SerializeToString().hex(),'grid':grid_from_obs(obs.text_data)}
class Context:
    def abort(self,code,message):raise RuntimeError(f'Official servicer abort: {code} {message}')
CTX=Context()

def config(env,seed):
    return {'env_name':env,'seed':str(seed),'data_dir':str(DATA),'render_mode':'text',
      'logging_path':str(ROOT/'raw/environment_logs'),'stack_frames':'false','skip_frames':'false','max_interaction_steps':'20'}

def new_path(env,seed,path,metered):
    official.Interpreter=(lambda:Meter(CURRENT)) if metered else native.Interpreter
    if path=='composite':
        service=interfaces.MARACompositeAutumnChangeDetectionServicer()
        init=service.Initialize(spb.InitializeRequest(config=config(env,seed)),CTX)
        reset=service.Reset(spb.ResetRequest(),CTX)
        base=service.interactive_environment.environment
        emit({'scenario':CURRENT,'event':'official_Initialize_Reset','init_hex':init.SerializeToString().hex(),
              'base_initial_observation':obs_record(reset.initial_observation),'base_has_prog':hasattr(base,'prog')})
        response=service.Step(spb.StepRequest(action=pb.Action(text_data='go-to-test')),CTX)
        assert service.transiting=='Change' and not response.is_terminal
        ui=service.change_detection_environment.environment
        initial=response.observation
        def available():
            q=spb.SpaceQueryRequest();q.reactive_query.SetInParent()
            return list(service.QuerySpaces(q,CTX).reactive_response.action_space.available_actions)
        def step(a):
            result=service.Step(spb.StepRequest(action=pb.Action(text_data=a)),CTX)
            if not result.HasField('observation'):raise RuntimeError('Wrapper returned no observation; possible swallowed exception')
            return result.observation,result.reward,result.is_terminal,dict(result.info)
    elif path=='direct':
        ui=official.CDSliderEnvironment(env,data_dir=str(DATA),seed=seed,
             logging_path=str(ROOT/'raw/environment_logs'))
        ui.reset();initial=ui.get_observation()
        available=ui.get_action_space
        def step(a):return ui.step(pb.Action(text_data=a))
    else:raise ValueError(path)
    return ui,initial,available,step

def run_scenario(env,seed,rep,rule,browses,path='composite',metered=True,suffix=''):
    global CURRENT
    prefix=P['primary_world_actions'] if env=='27VWC' else P['conditional_extension']['world_actions']
    CURRENT=f'{env}/seed{seed}/rep{rep}/{path}/{rule}/browse{browses}/{"meter" if metered else "native"}{suffix}'
    ui,initial,available,step=new_path(env,seed,path,metered)
    records=[];world=[grid_from_obs(initial.text_data)];score_calls=[]
    init_state=state(ui)
    assert init_state['time']==0 and init_state['frame_count']==1 and init_state['seed']==seed
    emit({'scenario':CURRENT,'event':'changed_initial_observation','observation':obs_record(initial),'state':init_state})
    reward_method=official.CDSliderEnvironment.change_reward
    def observed_reward():
        inputs=state(ui);reward=reward_method(ui)
        entry={'scorer_inputs':inputs,'reward':reward};score_calls.append(entry)
        emit({'scenario':CURRENT,'event':'official_reward_call',**entry})
        return reward
    ui.change_reward=observed_reward # Exact delegation; unbound original method used for the counterfactual.
    def act(a,kind):
        spaces=[x.text_data for x in available()];before=state(ui)
        ok=a in spaces or (a.startswith('click ') and any(s.startswith('click [') for s in spaces))
        request={'scenario':CURRENT,'event':'action_request','action':a,'kind':kind,'supported_actions':spaces,'pre_state':before}
        emit(request)
        if not ok:raise RuntimeError('Unadvertised action '+a)
        try:
            obs,reward,terminal,info=step(a)
            row={'action':a,'kind':kind,'pre_state':before,'post_state':state(ui),
                 'observation':obs_record(obs),'reward':reward,'terminal':terminal,'info':info}
            emit({'scenario':CURRENT,'event':'action_return',**row}) # Always persist BEFORE any assertion.
        except BaseException as e:
            emit({'scenario':CURRENT,'event':'action_exception','action':a,'error':repr(e),'traceback':traceback.format_exc(),'post_state':state(ui)})
            raise
        records.append(row)
        return row
    for a in prefix:
        row=act(a,'world');world.append(row['observation']['grid']);assert not row['terminal']
    guard=ui.trigger_start_time
    if guard is None:
        return {'scenario':CURRENT,'environment':env,'seed':seed,'rep':rep,'status':'NO_TRIGGER_FROZEN_PREFIX',
                'path':path,'rule':rule,'browses':browses,'world_grids':world,'records':records}
    act('I found the change!','declare')
    selected=None
    if rule!='none':
        selected={'guard-2':guard-2,'guard-1':guard-1,'guard':guard,'last':len(ui.frames)-1,'observed-2':4,'observed-1':5,'observed':6,'observed+1':7}[rule]
        if not 0<=selected<len(ui.frames):
            return {'scenario':CURRENT,'environment':env,'seed':seed,'rep':rep,'status':'FRAME_UNAVAILABLE',
                    'rule':rule,'requested_frame':selected,'guard':guard,'frame_count':len(ui.frames),'records':records}
        for _ in range(browses):act('choose_frame_'+str(selected),'select')
    pre_submit=state(ui)
    result=act('Submit choice','submit')
    assert len(score_calls)==1 and result['terminal']
    before_cf=state(ui);native_calls=sum(CALLS.values())
    cf_score=None;cf_inputs=None;paper_guard_proxy=None
    if selected is not None:
        clone=copy.copy(ui);clone.curr_frame=selected
        cf_inputs=state(clone)
        cf_score=reward_method(clone) # Unchanged public function, changed index ONLY, no changed official instance.
        if selected<guard-1:paper_guard_proxy=0
        elif selected in (guard-1,guard):paper_guard_proxy=1
        else:paper_guard_proxy=1.377/(1-(selected/guard)*math.exp(-selected/guard))-1.178
    assert state(ui)==before_cf and sum(CALLS.values())==native_calls
    row={'scenario':CURRENT,'environment':env,'seed':seed,'rep':rep,'status':'COMPLETE','path':path,
         'metered':metered,'rule':rule,'browses':browses,'prefix':prefix,'guard_time':guard,
         'selected_frame':selected,'initial_state':init_state,'pre_submit_state':pre_submit,
         'official_scorer_inputs':score_calls[0]['scorer_inputs'],'official_reward':result['reward'],
         'counterfactual_index_only_inputs':cf_inputs,'counterfactual_index_only_reward':cf_score,
         'independent_paper_formula_public_guard_proxy':paper_guard_proxy,
         'counterfactual_added_native_calls':0,'official_state_unchanged_by_counterfactual':True,
         'world_grids':world,'world_grid_digest':digest(world),'records':records}
    emit({'scenario':CURRENT,'event':'counterfactual','official_reward':result['reward'],
          'original_inputs':score_calls[0]['scorer_inputs'],'counterfactual_inputs':cf_inputs,
          'counterfactual_reward':cf_score,'guard_proxy_paper_score':paper_guard_proxy,
          'native_calls_added':0,'official_instance_unchanged':True})
    print('SCENARIO',CURRENT,'raw',result['reward'],'cf',cf_score,flush=True)
    return row

def semantic_returns(row):
    return {'world':row['world_grids'],'records':[{'action':x['action'],'observation':x['observation'],
       'reward':x['reward'],'terminal':x['terminal'],'info':x['info'],'pre':x['pre_state'],'post':x['post_state']}
       for x in row['records']]}

def factorial(env):
    rows=[]
    for seed in P['seeds']:
        for rep in range(P['repetitions']):
            for rule in ['guard-2','guard-1','guard','last']:
                for n in P['scoring_factorial']['selection_repetition_counts']:
                    rows.append(run_scenario(env,seed,rep,rule,n))
            rows.append(run_scenario(env,seed,rep,'none',0))
    return rows

COLOR=yaml.safe_load((DATA/'color_dict.yaml').read_text());INV={v:k for k,v in COLOR.items()}
def native_grid(engine):
    return env_utils.render_grid(json.loads(engine.render_all()),background_color=engine.get_background(),color_dict=INV)
def paired_replay(env,seed,rep):
    global CURRENT
    CURRENT=f'paired_replay/{env}/seed{seed}/rep{rep}'
    prefix=P['primary_world_actions'][:6] if env=='27VWC' else P['conditional_extension']['world_actions']
    actions=prefix+['noop']*(P['paired_original_changed_replay']['horizon']-len(prefix))
    original=(DATA/'programs'/f'{env}.sexp').read_text()
    changed=(DATA/'programs'/f'{env}_change_detection_wrong_program.sexp').read_text()
    event=json.loads((DATA/'answers'/f'{env}_change_detection.json').read_text())['condition']
    a=Meter(CURRENT+'/original');b=Meter(CURRENT+'/changed')
    ra=a.run_script(original,autumnstdlib,'',seed);rb=b.run_script(changed,autumnstdlib,event,seed)
    assert ra=='Script executed successfully.' and rb=='Script executed successfully.'
    ga=[native_grid(a)];gb=[native_grid(b)];trigger=None;guard_history=[]
    for t,action in enumerate(actions,1):
        before='true' in b.evaluate_to_string(event)
        if before and trigger is None:trigger=t
        assert env_utils.interpreter_action_to_text(a,action)
        assert env_utils.interpreter_action_to_text(b,action)
        middle='true' in b.evaluate_to_string(event)
        if middle and trigger is None:trigger=t
        a.step();b.step()
        ga.append(native_grid(a));gb.append(native_grid(b))
        guard_history.append({'step':t,'before_action':before,'after_input_before_transition':middle,'first_guard_time':trigger})
    diffs=[t for t,(x,y) in enumerate(zip(ga,gb)) if x!=y]
    row={'environment':env,'seed':seed,'rep':rep,'actions':actions,'horizon':len(actions),
         'original_program_sha256':hashlib.sha256(original.encode()).hexdigest(),
         'changed_program_sha256':hashlib.sha256(changed.encode()).hexdigest(),
         'programs_contain_random_token':{ 'original':'random' in original.lower() or 'uniform' in original.lower(),
                                          'changed':'random' in changed.lower() or 'uniform' in changed.lower()},
         'guard_expression':event,'guard_time':trigger,'first_observation_grid_divergence':diffs[0] if diffs else None,
         'differing_grid_times':diffs,'guard_history':guard_history,'original_grids':ga,'changed_grids':gb,
         'original_trajectory_digest':digest(ga),'changed_trajectory_digest':digest(gb)}
    emit({'scenario':CURRENT,'event':'paired_replay_complete','first_divergence':row['first_observation_grid_divergence'],'guard_time':trigger})
    print('PAIRED',env,seed,rep,'guard',trigger,'first_grid_difference',row['first_observation_grid_divergence'],flush=True)
    return row

result={'stage':'support','status':'running','observed_results':[],'failures':[],
        'independent_observable_target':6,'no_model_calls':True,'no_program_mutations':True}
started=time.monotonic()
try:
    prior=json.loads((ROOT/'results/extension_results.json').read_text())
    assert prior['status']=='completed'
    assert all(q['first_observation_grid_divergence']==6 for q in prior['paired_replays'])
    for seed in P['seeds']:
        for rep in range(P['repetitions']):
            for rule in ['observed-2','observed-1','observed','observed+1']:
                for n in [1,4]:
                    row=run_scenario('7WWW9',seed,rep,rule,n)
                    t=row['selected_frame'];target=6
                    row['independent_paper_formula_observation_target']=0 if t<target-1 else 1 if t in (target-1,target) else 1.377/(1-t/target*math.exp(-t/target))-1.178
                    result['observed_results'].append(row)
    result['status']='completed'
except BaseException as e:
    result['status']='failed';result['failures'].append({'error':repr(e),'traceback':traceback.format_exc()})
    print(traceback.format_exc(),flush=True)
finally:
    result['scientific_script_wall_seconds']=time.monotonic()-started
    result['native_call_counts']=dict(CALLS);result['total_metered_native_calls']=sum(CALLS.values())
    with OUTFILE.open('x') as f:json.dump(result,f,indent=2)
    print('SUPPORT_FINAL',result['status'],'calls',sum(CALLS.values()),flush=True)
    CALLLOG.close();TRACE.close()
sys.exit(0 if result['status']=='completed' else 2)
