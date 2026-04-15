invoke()
  ↓
stream()
  ↓
SyncPregelLoop (关键入口)
  ↓
PregelRunner
  ↓
loop.tick()


PregelRunner
    ↓
run_with_retry
    ↓
task.proc.invoke   ⭐⭐⭐（关键入口）
    ↓
node logic
    ↓
return / writes
    ↓
commit
    ↓
put_writes
    ↓
checkpoint