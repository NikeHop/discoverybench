from litellm import model_cost


def get_cost(llm, response):
    cost_dict = model_cost[llm.model_name]
    token_usage = response.usage_metadata
    
    cost = (
        token_usage["input_tokens"] * cost_dict["input_cost_per_token"]
        + token_usage["output_tokens"] * cost_dict["output_cost_per_token"]
    )

    return cost
