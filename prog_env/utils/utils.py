import json
from func_timeout import func_set_timeout, FunctionTimedOut
import contextlib
import io
from hashlib import sha256
from time import sleep
from openai import OpenAI
buf = io.StringIO()
@func_set_timeout(10)
def call_function(name, arguments, code, **kwargs):
    namespace = {}
    with contextlib.redirect_stdout(buf):
        exec(code, namespace, namespace)

    if name in namespace:
        predict = namespace[name](**arguments, **kwargs)
    else:
        raise NameError(f"name {name} is not defined")
    if type(predict) == dict or type(predict) == list:
        predict = json.dumps(predict, ensure_ascii=False)
    elif type(predict) != str:
        predict = str(predict)
    return predict
def answer_verify(predict, golden):
    golden = golden.split(', ')
    if type(predict) == dict or type(predict) == list:
        predict = json.dumps(predict, ensure_ascii=False)
    elif type(predict) != str:
        predict = str(predict)
    predict = predict.lower().replace(',', '').strip()

    for item in golden:
        item = item.lower().replace(',', '').strip()
        if item not in predict:
            return False

    return True
def get_feedback(tool_calls, codes, **kwargs):
    res = []
    if isinstance(tool_calls, dict):
        try:
            tool_name = tool_calls['name']
            tool_args = tool_calls['parameters']
            code = codes[tool_name]

            feedback = call_function(
                tool_name, tool_args, code, **kwargs)

            res.append({"role": "tool", "content": feedback})
        except FunctionTimedOut as e:
            res.append(
                {"role": "tool", "content": f"an error occured when call {tool_calls['name']}: {str(e)}"})
        except Exception as e:
            res.append(
                {"role": "tool", "content": f"an error occured when call {tool_calls['name']}: {str(e)}"})
    else:
        for tool_call in tool_calls:
            try:
                tool_name = tool_call['function']['name']
                tool_args = json.loads(tool_call['function']['arguments'])
                code = codes[tool_name]

                feedback = call_function(
                    tool_name, tool_args, code, **kwargs)

                res.append({"role": "tool", "content": feedback})
            except FunctionTimedOut as e:
                res.append(
                    {"role": "tool", "content": f"an error occured when call {tool_call['function']['name']}: {str(e)}"})
            except Exception as e:
                res.append(
                    {"role": "tool", "content": f"an error occured when call {tool_call['function']['name']}: {str(e)}"})

    return res

def get_params(model, args):
    if args.enable_thinking:
        max_new_tokens = 8192
    else:
        max_new_tokens = 1024

    if model == "qwen":
        return dict(do_sample=args.do_sample, max_new_tokens=max_new_tokens, temperature=args.temperature,
                    eos_token_id=[151645, 151643], pad_token_id=args.tokenizer.pad_token_id, use_cache=True)
    elif model == "llama31":
        return dict(do_sample=args.do_sample, max_new_tokens=max_new_tokens, temperature=args.temperature,
                    eos_token_id=[128001, 128008, 128009], pad_token_id=args.tokenizer.pad_token_id, use_cache=True)
    elif model in ["claude", "gemini"]:
        return dict(max_tokens=max_new_tokens, temperature=args.temperature)
    elif model in ["gpt"]:
        return dict(max_tokens=max_new_tokens, temperature=args.temperature)
    else:
        raise NotImplementedError
    

def chat_close(messages, args, tools=None, one_tool_only=False, base_url=None, api_keys=None, model=None, **kwargs):
    def _req_closed():
        try:
            logid = sha256(json.dumps(
                messages[0]["content"]).encode('utf-8')).hexdigest()
            
            call_model = model if model else args.model_path

            if not api_keys:
                raise ValueError("No API keys provided.")

            last_exception = None
            while True:
                for idx, key in enumerate(api_keys):
                    try:
                        print(f"Trying API key {idx + 1}/{len(api_keys)}...")
                        client = OpenAI(  
                            base_url = args.user_base_url,
                            api_key=key
                            )
                        response = client.chat.completions.create(
                            model=call_model,
                            messages=messages,
                            tools=tools,
                            temperature=args.temperature
                        )
                        return response.json()
                    except Exception as e:
                        last_exception = e
                        print(f"API key {idx + 1} failed: {e}")
                        if "Error code: 429" in str(e):
                            api_keys.remove(key)
                        sleep(2)
                        continue
                raise last_exception
        except Exception as e:
            print(
                f"Warning: There was a promblem when calling close api for messages:\n{messages}\nPass for next.\n{e}")
            return None

    message = {"role": "assistant", "content": ""}
    response = _req_closed()
    response = json.loads(response)
    if response is not None:
        if response.get("choices"):
            choice = response["choices"][0]
            for key, value in choice["message"].items():
                if value != "":
                    message[key] = value
            if message.get("tool_calls", None):
                if one_tool_only:
                    message["tool_calls"] = message["tool_calls"][:1]
                for tool_call in message["tool_calls"]:
                    if tool_call["function"].get("arguments"):
                        if json.loads(tool_call["function"]["arguments"]) is None:
                            tool_call["function"]["arguments"] = "{}"
                    else:
                        tool_call["function"]["arguments"] = "{}"
        elif "400: Invalid JSON payload received." in response.get("error", {}).get("message", "") or "400: Unable" in response.get("error", {}).get("message", ""):
            return message, api_keys
        elif "400: Invalid value at" in response.get("error", {}).get("message", ""):
            return message, api_keys
        elif "The server had an error processing your request." in response.get("error", {}).get("message", ""):
            return message, api_keys
        elif "is already defined at" in response.get("error", {}).get("message", ""):
            return message, api_keys
        elif response.get("StopReason", None) == "end_turn":
            return {"role": "assistant", "content": "Sorry, I'm temporarily unable to answer this question."}
        else:
            print(
                f"Warning: There was a promblem when calling close api for messages:\n{messages}\nPass for next.\n{response}")
            return None, api_keys
    return message, api_keys