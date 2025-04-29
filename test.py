import json 

from discovery_b.eval.new_eval import run_eval_gold_vs_gen_NL_hypo_workflow

from langchain.chat_models.base import init_chat_model
with open("test.json", "r") as f:
    data = json.load(f)

llm = init_chat_model(model="gpt-4o-mini")
data["llm"] = llm
run_eval_gold_vs_gen_NL_hypo_workflow(**data)