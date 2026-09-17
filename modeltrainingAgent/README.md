# Model Training Agent

Recoverable training orchestration and hyperparameter optimization using LangGraph,
Optuna, a replaceable trainer adapter, and a deterministic mock trainer that runs
without GPU, network, or LLM access.

## Quickstart

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
training-agent run --config configs/example.yaml
training-agent status --task-id demo-hpo-001
training-agent trials --task-id demo-hpo-001
training-agent report --task-id demo-hpo-001
```

The example creates three Optuna trials, writes checkpoints under
`artifacts/demo-hpo-001/`, and emits `artifacts/demo-hpo-001/final/report.json`.

## Design Notes

The graph treats Study, Trial, Attempt, and Epoch as separate concepts. Optuna is
the optimization source of truth. The Trial Repository is the execution and audit
source of truth. Checkpoints and reports live in the Artifact Store, and only paths
or small summaries are written to LangGraph State.

The first implementation ships a deterministic mock trainer. To connect a real
PyTorch or Transformers trainer, implement `TrainerAdapter` from
`training_agent.models` and return the same `TrainingResult` and
`EvaluationResult` shapes. The adapter must write checkpoints containing the trial
number and parameter hash, and must reject resume attempts when the hash differs.

## Recovery

The implementation uses stable request IDs for Optuna `ask`, immutable trial
parameter snapshots in SQLite, and idempotent artifact writes via temporary files
followed by atomic replace. This gives at-least-once execution with reconciliation,
not impossible cross-system exactly-once semantics.

## Quality

```bash
pytest
ruff check .
mypy src/training_agent
```


## graph node design 
没有把 services/core 直接注册进 graph，在node里进行封装后，然后节点注册；
一个 node 可以封装多个 service/core 方法。不是所有 services/ 里的每个方法都单独对应一个 node，LangGraph 的 node 是一个业务步骤，不是一个底层函数一个 node

当前结构：
graph.py
  只注册 node

nodes/*.py
  接收完整 state
  读取关键字段
  调用 service/core
  返回关键字段更新
    
services/*.py
     提供真正核心能力


# Graph 节点说明

import_model_project_config
    - 读取 VLM 配置
    - 生成 Agent config
    - 提取 model_entrypoint/loss_entrypoint/optimizer_entrypoint
    - 生成 search_space
    - 写入 state

load_dataset
    - 读取 dataset_uri
    - 检查文件是否存在
    - 检查 JSONL 格式
    - 统计样本数、字段、label 候选
    - 计算 dataset_version/hash

classify_task
    - 如果用户配置了 task_type，直接用
    - 否则根据 dataset_summary 推断
    -validate_inputs
        检查任务需要的字段是否存在
        检查样本数量是否足够
        检查 objective 是否支持
        如果失败，进入failure_report

prepare_runtime
    -准备运行环境，只做一次。
    -创建/加载 Optuna Study
    -创建 artifact 目录
    -写 task_config.json
    -写 dataset_manifest.json
    -固定 train/validation/test 数据引用
    
suggest_trial
    让 Optuna 生成一组超参数。
    生成稳定 request_id
    先查 repository 是否已有这个 request_id
    如果没有，再调用 Optuna ask()
    生成 trial_params
    生成 runtime_params
    初始化当前 Trial 状态

validate_trial
    验证这一组 Trial 参数能不能跑。
    检查 batch/runtime 参数关系
    调用 trainer.validate()
    发现非法配置则标记 Trial failed

persist_trial
    把 Trial 配置固化保存。
    计算 trial_params hash
    写 TrialRecord 到 repository
    创建 attempt 记录
    写 trial_config.json
    关键点：
    trial_params 一旦保存，不允许静默覆盖

run_training
    构造 TrainingContext
    调用 LocalExecutor
    LocalExecutor 调 trainer.train() 或 trainer.resume()
    拿到 checkpoint、metrics、状态
    写 execution 记录
    如果成功：
    execution_status = completed
    如果中断：
    execution_status = interrupted
    如果 OOM：
    execution_status = recoverable_error
    error_type = oom


submit_job
    远程训练提交节点。
    构造 RemoteTrainingRequest
    调用 remote.submit()
    拿到 external_job_id


monitor_job
    程训练监控节点。
    用 external_job_id 查询远程任务状态
    如果还没完成，返回 running/submitted
    如果完成，拿训练结果
    写 execution 记录
    输出类似 run_training：
    execution_status
    latest_metrics
    checkpoint_uri
    error_type

resume_trial
    同一个 Trial 中断恢复。
    触发条件：
    execution_status = interrupted
    检查 trial_checkpoint_uri
    trial_attempt + 1
    创建新的 attempt 记录
    不创建新 Trial
    不改变 trial_params
    然后回到：
    run_training 还是同一个 trial_number


repair_trial
    可恢复错误修复，例如 OOM。
    触发条件：
    execution_status = recoverable_error
    如果是 OOM：
    降低 micro_batch_size
    增加 gradient_accumulation_steps
    保持 trial_params 不变
    trial_attempt + 1
    然后回到：validate_trial
    关键点：
    只改 runtime_params
    不改 learning_rate/batch_size/weight_decay 这些 HPO 参数

evaluate_trial
    验证集评估。
    加载当前 Trial 最佳 checkpoint
    调用 trainer.evaluate()
    拿 objective 指标
    保存 metrics


complete_optuna_trial
    告诉 Optuna 这个 Trial 的结果。
    study.tell(trial_number, objective_value)
    repository.mark_completed()
    查询 best_trial
    更新 completed/failed/pruned 计数
    判断是否应该停止
    清空当前 active trial 字段


diagnose_trial
    练诊断。
    调用 diagnosis service
    生成结构化诊断
    当前默认是 NoOp：
    不会影响主流程


finalize
    终成功收尾。
    触发条件：
    达到目标指标
    达到最大 Trial 数
    达到时间预算
    达到 patience
    查询最佳 Trial
    检查最佳 checkpoint
    只在这里做 test set 最终评估
    写 best_model.ref.json
    写 report.json
    设置 workflow_status = completed


failure_report
    失败收尾。
    触发条件：
    数据坏了
    配置坏了
    fatal error
    无法恢复
    写 failure_report.json
    记录失败节点、错误类型、错误信息、当前 Trial、checkpoint
    设置 workflow_status = failed




1. import_model_project_config
# 输入 state 关键字段：
fixed_training_config.model_project_config_uri
fixed_training_config.project_type
fixed_training_config.generated_task_id
fixed_training_config.agent_config_output_uri
fixed_training_config.entrypoint_module
task_id
# 输出更新字段：
updated_at
task_type
dataset_uri
base_checkpoint_uri
fixed_training_config
study_name
objective_name
objective_direction
search_space
max_trials
target_value
patience
min_delta
total_epochs
events



2. prepare_runtime


# 输入 state 关键字段：
task_id
dataset_uri
dataset_version
# 输出更新字段：
updated_at
workflow_status
train_dataset_uri
validation_dataset_uri
test_dataset_uri
events



3. suggest_trial


输入 state 关键字段：
task_id
trial_number
trial_request_id
completed_trial_count
failed_trial_count
pruned_trial_count
search_space

输出更新字段：
updated_at
trial_request_id
trial_number
trial_status
trial_params
runtime_params
trial_attempt
execution_status
external_job_id
current_epoch
trial_checkpoint_uri
trial_best_checkpoint_uri
latest_metrics
trial_best_metrics
objective_value
error_type
error_message
error_node
retryable
events



4. validate_trial

输入 state 关键字段：
trial_number
trial_params
runtime_params
task_id
输出更新字段：
成功：
updated_at
runtime_params
trial_status
events
失败：
trial_status
execution_status
error_type
error_message
error_node
stop_reason
should_stop
events

5. persist_trial
调用的 service 出口：
services.repository.create_or_get_trial()
services.repository.start_attempt()
services.artifacts.write_json()
输入 state 关键字段：
task_id
trial_number
trial_request_id
trial_params
runtime_params
trial_attempt
trial_started_at
输出更新字段：
updated_at
trial_status
trial_started_at
events

6. run_training
调用的 service/执行器出口：
services.artifacts.attempt_dir()
LocalExecutor(services.trainer).run()
services.repository.update_execution()
输入 state 关键字段：
task_id
trial_number
trial_attempt
trial_params
runtime_params
base_checkpoint_uri
trial_checkpoint_uri
train_dataset_uri
validation_dataset_uri
dataset_uri
total_epochs
输出更新字段：
updated_at
execution_status
latest_metrics
trial_checkpoint_uri
trial_best_checkpoint_uri
current_epoch
training_duration_seconds
gpu_hours
error_type
error_message
error_node
retryable
events
结论：本地训练执行和执行记录更新已被 node 封装，node 已注册进 graph。
7. submit_job / monitor_job
调用的 service 出口：
services.remote.submit()
services.remote.status()
services.repository.update_execution()
输入 state 关键字段：
task_id
trial_number
trial_attempt
external_job_id
trial_params
runtime_params
trial_checkpoint_uri
输出更新字段：
submit_job：
updated_at
external_job_id
execution_status
events
monitor_job：
updated_at
execution_status
latest_metrics
trial_checkpoint_uri
trial_best_checkpoint_uri
current_epoch
training_duration_seconds
gpu_hours
error_type
error_message
error_node
retryable
events
结论：远程执行器已通过 node 封装，node 已注册进 graph。
8. resume_trial / repair_trial
调用的 service 出口：
services.repository.start_attempt()
services.optuna.fail()
services.repository.mark_failed()
输入 state 关键字段：
trial_checkpoint_uri
trial_attempt
retry_count
max_retries
error_type
error_message
runtime_params
trial_params
trial_number
task_id
输出更新字段：
trial_attempt
runtime_params
execution_status
retry_count
error_type/error_message/stop_reason/should_stop
events
结论：恢复和 OOM 修复已封装成 node，并注册进 graph。
9. evaluate_trial
调用的 service 出口：
services.trainer.evaluate()
services.repository.save_metrics()
输入 state 关键字段：
execution_status
trial_best_checkpoint_uri
task_id
trial_number
validation_dataset_uri
dataset_uri
objective_name
输出更新字段：
updated_at
objective_value
trial_best_metrics
events
结论：评估函数已被 node 封装，node 已注册进 graph。
10. complete_optuna_trial
调用的 service 出口：
services.optuna.complete()
services.repository.mark_completed()
services.optuna.best_trial()
services.optuna.counts()
输入 state 关键字段：
trial_number
objective_value
task_id
objective_direction
min_delta
best_objective_value
trial_best_metrics
trial_best_checkpoint_uri
elapsed_seconds
training_duration_seconds
输出更新字段：
updated_at
completed_trial_count
failed_trial_count
pruned_trial_count
no_improvement_count
elapsed_seconds
trial_finished_at
best_trial_number
best_objective_value
best_params
best_metrics
best_checkpoint_uri
should_stop
stop_reason
trial_request_id
trial_number
trial_status
trial_params
runtime_params
trial_attempt
execution_status
external_job_id
events
结论：Optuna complete 已被 node 封装，node 已注册进 graph。
11. diagnose_trial
调用的 service 出口：
services.diagnosis.diagnose()
输入 state 关键字段：
理论上传整个 state
输出更新字段：
updated_at
diagnosis
diagnosis_status
events
结论：诊断服务已被 node 封装，node 已注册进 graph。
12. finalize / failure_report
调用的 service 出口：
services.optuna.best_trial()
services.trainer.evaluate()
services.artifacts.final_dir()
services.artifacts.write_json()
services.optuna.summaries()
services.repository.set_final_report()
输入 state 关键字段：
task_id
best_checkpoint_uri
test_dataset_uri
validation_dataset_uri
dataset_uri
objective_name
task_type
dataset_version
study_name
objective_direction
search_space
search_space_version
best_metrics
elapsed_seconds
gpu_hours
completed_trial_count
failed_trial_count
pruned_trial_count
stop_reason
输出更新字段：
finalize：
updated_at
workflow_status
final_model_uri
final_report_uri
events
failure_report：
updated_at
workflow_status
final_report_uri
should_stop
stop_reason
events
结论：最终报告和失败报告已封装成 node，并注册进 graph。
需要特别说明的地方




services/entrypoint_loader.py
通用核心能力，不属于某个 node，入口解析器，用来把字符串变成真正函数。如：
load_entrypoint("python://vlm_project.adapters.training_agent_entrypoints:build_model")


trainers/vlm.py
是一个 TrainerAdapter 实现；

graph.py
