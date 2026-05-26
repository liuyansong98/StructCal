import numpy as np
import requests
import json
import os
import re

def main():
    # === 1. 服务地址配置（按你启动时的 args.host / args.port 改） ===
    host = "127.0.0.1"   # 如果你是 args.host="0.0.0.0"，客户端用 127.0.0.1 即可
    port = 5001          # 改成你实际 args.port，例如 9000
    url = f"http://{host}:{port}/reward"

    # === 2. 构造要发送的 JSON 数据 ===
    # idx_list: 你要查询的样本索引列表（注意要在 [0, len(heads)) 范围内）
    # 这里只是示例，假设数据集规模足够大，先用 0,1,2 三个样本测试
    idx_list = [60000, 60001, 60002]

    # 测试TKGR_server.py
    # selected_path_list: 和 idx_list 一一对应的路径字符串列表
    # 如果先不想管路径解析，*可以先传空字符串*，模型照样能跑（只是不用路径特征）
    # selected_path_list = [
    #     "",
    #     "",
    #     ""
    # ]
    # selected_path_list = [
    #     "1. Food and Agriculture Organization -> Provide aid(242) -> Ministry (Vietnam);\n"
    #     "Food and Agriculture Organization -> Express intent to meet or negotiate(92) -> North Korea;\n"
    #     "Food and Agriculture Organization -> Consult(88) -> Afonso Pedro Canga;\n",
    #     "1. Japan -> Express intent to meet or negotiate(247) -> South Korea;\n"
    #     "2. Japan -> Express intent to meet or negotiate(247) -> China;\n"
    #     "3. Japan -> INV::Express intent to meet or negotiate(247) -> South Korea;\n"
    #     "4. Japan -> Host a visit(247) -> Association of Southeast Asian Nations;\n"
    #     "5. Japan -> INV::Express intent to engage in diplomatic cooperation (such as policy support)(247) -> Thailand;\n",
    #     " Martin Lidegaard -> Express intent to meet or negotiate(243) -> Iran;\n"
    #     "Martin Lidegaard -> Express intent to meet or negotiate(243) -> Mohammad Javad Zarif;\n"
    # ]
    # payload = {
    #     "idx_list": idx_list,
    #     "selected_path_list": selected_path_list
    #     # 如果你后面在 deal_request 里加了别的参数（kwargs），可以一并塞进来
    # }

    # 测试reward.py
    payload = {
        "query_list" : ["""<path_list>
Police (South Africa)->Confiscate property(6)->Arrest, detain, or charge with legal action(0)->Men (South Africa);0.008
Police (South Africa)->Confiscate property(6)->Arrest, detain, or charge with legal action(1)->Men (South Africa);0.008
Police (South Africa)->Confiscate property(6)->INV::Sign formal agreement(5)->Government (Angola);0.008
Police (South Africa)->Confiscate property(6)->INV::Confiscate property(6)->Police (South Africa);0.008
Police (South Africa)->Confiscate property(6)->Arrest, detain, or charge with legal action(5)->Men (South Africa);0.008
Police (South Africa)->Confiscate property(6)->Arrest, detain, or charge with legal action(4)->Men (South Africa);0.008
Police (South Africa)->Arrest, detain, or charge with legal action(16)->INV::Arrest, detain, or charge with legal action(14)->Police (South Africa);0.008
Police (South Africa)->Arrest, detain, or charge with legal action(16)->INV::Use conventional military force(13)->Police (South Africa);0.008
Police (South Africa)->Arrest, detain, or charge with legal action(16)->Abduct, hijack, or take hostage(13)->Businessperson (South Africa);0.008
Police (South Africa)->Arrest, detain, or charge with legal action(23)->INV::Arrest, detain, or charge with legal action(1)->South Africa;0.008
</path_list>
The rank of the true tail entity: 1.0/7128

<selected_path_list>
Police (South Africa)->Confiscate property(6)->Arrest, detain, or charge with legal action(1)->Men (South Africa);0.008
Police (South Africa)->Confiscate property(6)->INV::Sign formal agreement(5)->Government (Angola);0.008
Police (South Africa)->Confiscate property(6)->INV::Confiscate property(6)->Police (South Africa);0.008
Police (South Africa)->Confiscate property(6)->Arrest, detain, or charge with legal action(5)->Men (South Africa);0.008
</selected_path_list>

<path_list>
Police (South Africa)->Arrest, detain, or charge with legal action(15)->INV::Arrest, detain, or charge with legal action(6)->Police (South Africa);0.008
Police (South Africa)->Arrest, detain, or charge with legal action(15)->INV::Arrest, detain, or charge with legal action(7)->Police (South Africa);0.008
Police (South Africa)->Arrest, detain, or charge with legal action(15)->INV::Arrest, detain, or charge with legal action(5)->South Africa;0.008
Police (South Africa)->Arrest, detain, or charge with legal action(15)->INV::Arrest, detain, or charge with legal action(8)->South Africa;0.008
Police (South Africa)->Arrest, detain, or charge with legal action(15)->INV::Arrest, detain, or charge with legal action(15)->Police (South Africa);0.008
Police (South Africa)->INV::Praise or endorse(22)->Praise or endorse(22)->Police (South Africa);0.008
Police (South Africa)->INV::Praise or endorse(22)->INV::Abduct, hijack, or take hostage(13)->Men (South Africa);0.008
Police (South Africa)->INV::Make statement(23)->Make statement(23)->Police (South Africa);0.008
Police (South Africa)->Arrest, detain, or charge with legal action(15)->INV::Use conventional military force(15)->Police (South Africa);0.008
Police (South Africa)->INV::Threaten with military force(27)->INV::Arrest, detain, or charge with legal action(22)->South Africa;0.008
</path_list>
The rank of the true tail entity: 1.0/7128

<selected_path_list>
Police (South Africa)->INV::Make statement(23)->Make statement(23)->Police (South Africa);0.008
Police (South Africa)->Arrest, detain, or charge with legal action(15)->INV::Use conventional military force(15)->Police (South Africa);0.008
Police (South Africa)->INV::Threaten with military force(27)->INV::Arrest, detain, or charge with legal action(22)->South Africa;0.008
</selected_path_list>

<path_list>
Police (South Africa)->Arrest, detain, or charge with legal action(4)->INV::Arrest, detain, or charge with legal action(4)->Police (South Africa);0.008
Police (South Africa)->INV::Demand economic aid(16)->INV::Arrest, detain, or charge with legal action(0)->South Africa;0.008
Police (South Africa)->Increase police alert status(16)->Arrest, detain, or charge with legal action(0)->Men (South Africa);0.008
Police (South Africa)->INV::Demand economic aid(16)->INV::Investigate(13)->Police (South Africa);0.008
Police (South Africa)->Increase police alert status(16)->Praise or endorse(13)->Education (South Africa);0.008
Police (South Africa)->fight with small arms and light weapons(23)->INV::Arrest, detain, or charge with legal action(23)->Police (South Africa);0.008
Police (South Africa)->fight with small arms and light weapons(23)->INV::Arrest, detain, or charge with legal action(13)->South Africa;0.008
Police (South Africa)->Increase police alert status(16)->INV::Make a visit(15)->Gambia;0.008
Police (South Africa)->INV::Demand intelligence cooperation(24)->Demand intelligence cooperation(24)->Police (South Africa);0.008
Police (South Africa)->Increase police alert status(16)->INV::Confiscate property(6)->Police (South Africa);0.008
</path_list>
The rank of the true tail entity: 1.0/7128

Maximum number of interaction rounds has been reached. Based on your background knowledge, the ~@~\History Entities~@~] provided at the beginning, and the interaction information above, predict the tail entity of the ~@~\Query~@~~
]. Wrap the results in <prediction_list></prediction_list>.
<prediction_list>
1. Moses Mathendele Dlamini:20
2. Men (South Africa):19
3. Police (South Africa):18
4. South Africa:17
5. Businessperson (South Africa):16
6. Government (Angola):15
7. Education (South Africa):14
8. Gambia:13
9. Men (South Africa):12
10. Murderer (South Africa):11
</prediction_list>
"""],
        "idx_list": ['647'],
        "current_step": 1,
        "recall_num_list": [2]
    }


    print("Request JSON:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    # === 3. 发送 POST 请求 ===
    try:
        resp = requests.post(url, json=payload, timeout=30)
    except requests.exceptions.RequestException as e:
        print("HTTP 请求失败:", e)
        return

    print("\nHTTP status code:", resp.status_code)

    # === 4. 解析返回 JSON ===
    try:
        data = resp.json()
    except ValueError:
        print("返回的不是合法 JSON：", resp.text)
        return

    # 按你服务端代码，返回结构是：
    # {"path_sets": path_list, "rank_list": rank_list, "entity_num": num_entities}
    print("\nResponse JSON:")
    print(json.dumps(data, ensure_ascii=False, indent=2))

    # 也可以分别取字段看看：
    path_sets = data.get("path_sets", [])
    rank_list = data.get("rank_list", [])
    entity_num = data.get("entity_num")

    print("\n解析后的字段：")
    print("entity_num:", entity_num)
    print("rank_list:", rank_list)
    print("path_sets 示例（第一个样本）：")
    for path_set in path_sets:
        print(path_set)

def test_tokenizer_encode():
    text = """
    \n\n<path_list>
    Foreign Affairs (United States)->INV::Host a visit(48)->Ministry (Jordan); 0.125
    Foreign Affairs (United States)->Make a visit(48)->Ministry (Jordan); 0.124
    Foreign Affairs (United States)->Make a visit(47)->North Korea; 0.057
    Foreign Affairs (United States)->INV::Host a visit(45)->China; 0.017
    Foreign Affairs (United States)->Make a visit(45)->China; 0.017
    Foreign Affairs (United States)->INV::Consult(44)->Yariv Levin; 0.015
    Foreign Affairs (United States)->Make a visit(44)->North Korea; 0.014
    Foreign Affairs (United States)->INV::Host a visit(42)->Japan; 0.009
    Foreign Affairs (United States)->Appeal for release of persons or property(44)->North Korea; 0.009
    Foreign Affairs (United States)->Criticize or denounce(44)->North Korea; 0.008
    </path_list>
    The rank of the true tail entity: 419.0/7128

    <selected_path_list>
    Foreign Affairs (United States)->Make a visit(47)->North Korea; 0.057
    Foreign Affairs (United States)->INV::Host a visit(45)->China; 0.017
    Foreign Affairs (United States)->Make a visit(45)->China; 0.017
    Foreign Affairs (United States)->INV::Consult(44)->Yariv Levin; 0.015
    </selected_path_list>

    <path_list>
    Foreign Affairs (United States)->INV::Host a visit(48)->Ministry (Jordan); 0.093
    Foreign Affairs (United States)->INV::Host a visit(47)->North Korea; 0.044
    Foreign Affairs (United States)->Make a visit(47)->North Korea; 0.043
    Foreign Affairs (United States)->INV::Host a visit(45)->China; 0.014
    Foreign Affairs (United States)->Make a visit(45)->China; 0.013
    Foreign Affairs (United States)->INV::Consult(44)->Yariv Levin; 0.012
    Foreign Affairs (United States)->Make a visit(44)->North Korea; 0.011
    Foreign Affairs (United States)->Make a visit(42)->Japan; 0.007
    Foreign Affairs (United States)->Make optimistic comment(44)->South Korea; 0.007
    Foreign Affairs (United States)->Make a visit(45)->INV::Express intent to meet or negotiate(44)->John Kerry;0.007
    </path_list>
    The rank of the true tail entity: 539.0/7128

    <selected_path_list>
    Foreign Affairs (United States)->INV::Host a visit(45)->China;
    Foreign Affairs (United States)->Make a visit(45)->China;
    Foreign Affairs (United States)->INV::Consult(44)->Yariv Levin;
    </selected_path_list>

    <path_list>
    Foreign Affairs (United States)->INV::Host a visit(48)->Ministry (Jordan); 0.08
    Foreign Affairs (United States)->Make a visit(48)->Ministry (Jordan); 0.079
    Foreign Affairs (United States)->INV::Host a visit(47)->North Korea; 0.038
    Foreign Affairs (United States)->Make a visit(47)->North Korea; 0.038
    Foreign Affairs (United States)->Make a visit(45)->China; 0.012
    Foreign Affairs (United States)->INV::Host a visit(44)->North Korea; 0.01
    Foreign Affairs (United States)->Make a visit(44)->North Korea; 0.01
    Foreign Affairs (United States)->INV::Host a visit(47)->Host a visit(44)->China;0.01
    Foreign Affairs (United States)->INV::Host a visit(47)->INV::Make statement(44)->South Korea;0.009
    Foreign Affairs (United States)->INV::Host a visit(47)->INV::Reject(43)->South Korea;0.009
    </path_list>
    The rank of the true tail entity: 374.0/7128

    Maximum number of interaction rounds has been reached. Based on your background knowledge, the \"History Entities\" provided at the beginning, and the interaction information above, predict the tail entity of the \"Query\". Wrap the results in <prediction_list></prediction_list>.\n"
    ]. Wrap the results in <prediction_list></prediction_list>.
    <prediction_list>
    1. Ministry (Jordan):10
    2. North Korea:9
    3. China:8
    4. Japan:7
    5. South Korea:6
    6. Afghanistan:5
    7. France:4
    8. Mevlut Cavusoglu:3
    9. Labaran Maku:2
    10. Aïchatou Mindaoudou Souleymane:1
    </prediction_list>
        """
    PATH_LIST_BEG = '<path_list>'
    PATH_LIST_END = '</path_list>'
    SELE_PATH_BEG = '<selected_path_list>'
    SELE_PATH_END = '</selected_path_list>'
    PRED_BEG = '<prediction_list>'
    PRED_END = '</prediction_list>'
    pred_add_str = f"Maximum number of interaction rounds has been reached. Based on your background knowledge, the \"History Entities\" provided at the beginning, and the interaction information above, predict the tail entity of the \"Query\". Wrap the results in {PRED_BEG}{PRED_END}.\n"
    # (1) 固定长句：允许行首有空白；必须以该句内容出现（含末尾 \n）
    pred_add_str_re = re.compile(r"^\s*" + re.escape(pred_add_str), flags=re.MULTILINE)  # 过滤规则。mask掉非llm生成的token
    beg = re.escape(PATH_LIST_BEG)
    end = re.escape(PATH_LIST_END)
    path_list_re = re.compile(
        rf"""
                        \n\n            # 你代码里是 \\n\\n，但这里兼容只有 \\n 或文本开头
                        \s*{beg}\s*\n            # <path_list> + \\n
                        .*?                      # path_block（跨行，非贪婪）
                        \n\s*{end}\s*\n          # </path_list> + \\n
                        \s*The\ rank\ of\ the\ true\ tail\ entity:\s*
                        \d+(?:\.\d+)?/\d+        # 3.0/7128 / 16/7128 / 16.0/7128
                        \n\n            # 末尾可能是 \\n\\n、\\n 或结束
                        """,
        flags=re.DOTALL | re.VERBOSE,
    )
    patterns = [pred_add_str_re, path_list_re]

    for pat in patterns:
        print(pat)
        for m in pat.finditer(text):
            ms, me = m.span()  # 匹配到的字符区间 [ms, me)
            # 映射回 token：与 [ms, me) 有交集的 token 全 mask
            print(ms, me)
    from transformers import AutoTokenizer

    # 替换为你脚本中的 BASE_PATH
    model_path = "../modelscope_cache/Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # 将文字转换为 token id 列表
    tokens = tokenizer.encode(text)
    print(f"Token 数量: {len(tokens)}")

    seq1 = '''
            <path_list>
            Other Authorities / Officials (Bahrain)->INV::Criticize or denounce(105)->Make a visit(83)->INV::Express intent to meet or negotiate(72)->Xi Jinping;0.009
            Other Authorities / Officials (Bahrain)->Arrest, detain, or charge with legal action(131)->INV::Arrest, detain, or charge with legal action(72)->Reduce or break diplomatic relations(72)->Qatar;0.009
            Other Authorities / Officials (Bahrain)->Arrest, detain, or charge with legal action(131)->INV::Arrest, detain, or charge with legal action(72)->Criticize or denounce(71)->Nuri al-Maliki;0.009
            Other Authorities / Officials (Bahrain)->Arrest, detain, or charge with legal action(131)->INV::Arrest, detain, or charge with legal action(131)->INV::Criticize or denounce(105)->International Government Organizations;0.009
            Other Authorities / Officials (Bahrain)->Investigate(248)->Defense Attorney (Bahrain); 0.009
            Other Authorities / Officials (Bahrain)->INV::Criticize or denounce(105)->Express intent to meet or negotiate(76)->INV::Make a visit(68)->Barack Obama;0.009
            Other Authorities / Officials (Bahrain)->INV::Make an appeal or request(62)->Make an appeal or request(21)->INV::Praise or endorse(19)->France;0.009
            Other Authorities / Officials (Bahrain)->Arrest, detain, or charge with legal action(131)->INV::Arrest, detain, or charge with legal action(131)->INV::Make an appeal or request(91)->Amnesty International;0.009
            Other Authorities / Officials (Bahrain)->INV::Make an appeal or request(62)->Make an appeal or request(21)->Sign formal agreement(16)->United Arab Emirates;0.009
            Other Authorities / Officials (Bahrain)->INV::Criticize or denounce(105)->Make a visit(83)->Host a visit(82)->Julie Bishop;0.009
            </path_list>\nThe rank of the true tail entity: 3.0/7128\n\n'''
    seq2 = "<path_list>\n"
    token_ids1 = tokenizer.encode(f"{seq1}", add_special_tokens=False)
    token_ids2 = tokenizer.encode(f"{seq2}", add_special_tokens=False)
    print(f"token_ids1: {token_ids1}")
    print(f"token_ids2: {token_ids2}")

if __name__ == "__main__":

    main()




