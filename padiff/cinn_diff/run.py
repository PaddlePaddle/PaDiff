import os
from analyze import auto_diff
from env import Env


def run(run_script, base_env, cinn_env):
    run_env = Env(run_script, base_env, cinn_env)
    run_env.run_base_model() #可以提供选项选择不运行base model
    run_env.run_cinn_model()
    auto_diff(run_env.base_path, run_env.cinn_path, rtol=0, atol=0)


if __name__ == '__main__':
    run_script = "/root/dev/PaddleNLP/model_zoo/bert/run_bert.sh"
    run(run_script, None, None)

