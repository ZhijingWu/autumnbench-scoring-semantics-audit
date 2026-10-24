import pathlib,json,subprocess,os,resource,time,signal,shutil,sys
R=pathlib.Path('external/local-resource')
stage=sys.argv[1];assert stage in ['support']
budget=R/'manifests/execution_budget.json'
prior=json.loads(budget.read_text()) if budget.exists() else {'runs':[]}
assert not any(x['stage']==stage for x in prior['runs'])
remaining=1800-sum(x['wall_seconds'] for x in prior['runs']);assert remaining>0
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
def cap():
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
start=time.monotonic();peak=0;failure=None;disk=0
log=R/'raw'/f'{stage}.process.log'
with log.open('x') as f:
    p=subprocess.Popen(['external/local-resource',str(R/'scripts/run_support_probe.py'),stage],
                       cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=cap)
    while p.poll() is None:
        rss=0
        for q in pathlib.Path('/proc').iterdir():
            if not q.name.isdigit():continue
            try:
                if os.getpgid(int(q.name))!=p.pid:continue
                rss+=next((int(s.split()[1])*1024 for s in (q/'status').read_text().splitlines() if s.startswith('VmRSS:')),0)
            except (OSError,ProcessLookupError):pass
        peak=max(peak,rss)
        disk=sum(q.stat().st_size for q in R.rglob('*') if q.is_file())
        if rss>8*1024**3:failure='group RSS cap'
        if disk>2*1024**3:failure='project disk cap'
        if time.monotonic()-start>remaining:failure='wall cap'
        if shutil.disk_usage(R).free<2*1024**3:failure='host free-disk reserve'
        if failure:os.killpg(p.pid,signal.SIGTERM);break
        time.sleep(.1)
    rc=p.wait(timeout=10)
row={'stage':stage,'returncode':rc,'wall_seconds':time.monotonic()-start,
     'peak_sampled_group_rss_bytes':peak,'project_bytes_last_sample':disk,'cap_failure':failure,'log':str(log)}
prior['runs'].append(row)
prior.update(total_scientific_process_wall_seconds=sum(x['wall_seconds'] for x in prior['runs']),
             scientific_wall_cap_seconds=1800,group_ram_cap_bytes=8*1024**3,project_disk_cap_bytes=2*1024**3,
             cpu_affinity=[0,1],gpu_used=False,model_calls=0)
budget.write_text(json.dumps(prior,indent=2))
print(json.dumps(row,indent=2));print(log.read_text()[-5500:]);sys.exit(rc)
