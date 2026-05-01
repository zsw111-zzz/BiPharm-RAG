# -*- coding: utf-8 -*-
"""TCM-specific prompt profile for the BiPharm-RAG local runtime."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .prompt import PROMPTS as ACTIVE_PROMPTS


_DEFAULT_PROMPTS_SNAPSHOT = deepcopy(ACTIVE_PROMPTS)


TCM_DEFAULT_ENTITY_TYPES = [
    "病名",
    "西医对应病",
    "证型",
    "病机",
    "脏腑",
    "病位",
    "病性",
    "症状",
    "体征",
    "舌质",
    "舌苔",
    "脉象",
    "治法",
    "治疗原则",
    "方剂",
    "中成药",
    "药材",
    "药味",
    "剂量",
    "疗程",
    "加减条件",
    "禁忌",
    "注意事项",
    "特殊人群",
    "生理时期",
    "体质",
    "检查",
    "指标",
    "穴位",
]


TCM_THEME_EXAMPLES = [
    """Example 1:

Text:
咳嗽痰黄，咯痰黏稠，口渴，咽痛，舌红，苔黄腻，脉滑数。病机为痰热壅肺，肺失宣肃。治法当清热肃肺，化痰止咳。可参考清金化痰汤加减。
################
Output:
("theme"{tuple_delimiter}本段核心主轴为痰热壅肺，治宜清热肃肺、化痰止咳，可参考清金化痰汤加减，适用于咳嗽痰黄、舌红苔黄腻、脉滑数等表现。){record_delimiter}
("key_entity"{tuple_delimiter}痰热壅肺{tuple_delimiter}病机键{tuple_delimiter}痰热壅肺是本段辨证主轴，决定后续治法与方药方向。{tuple_delimiter}96){record_delimiter}
("key_entity"{tuple_delimiter}清热肃肺{tuple_delimiter}治法键{tuple_delimiter}清热肃肺是围绕痰热壅肺所采用的核心治法。{tuple_delimiter}90){record_delimiter}
("key_entity"{tuple_delimiter}清金化痰汤{tuple_delimiter}方剂键{tuple_delimiter}清金化痰汤是与该病机和治法相匹配的代表方剂。{tuple_delimiter}88){record_delimiter}
("key_entity"{tuple_delimiter}咳嗽痰黄{tuple_delimiter}症状键{tuple_delimiter}咳嗽痰黄是支持痰热壅肺判断的重要临床表现。{tuple_delimiter}84){completion_delimiter}
#############################""",
]


TCM_ENTITY_EXAMPLES = [
    """Example 1:

Entity_types: [病名, 证型, 病机, 症状, 舌质, 舌苔, 脉象, 治法, 方剂]
Text:
咳嗽痰黄，痰稠难咯，口渴，舌红，苔黄腻，脉滑数。病机为痰热壅肺，肺失宣肃。治法当清热肃肺，化痰止咳。方药可参考清金化痰汤加减。
################
Output:
("Entity"{tuple_delimiter}咳嗽痰黄{tuple_delimiter}症状{tuple_delimiter}咳嗽痰黄提示肺失宣肃并夹痰热，是辨证的重要症状。{tuple_delimiter}表现: 痰黄, 痰稠, 咳嗽){record_delimiter}
("Entity"{tuple_delimiter}舌红{tuple_delimiter}舌质{tuple_delimiter}舌红反映里热偏盛。{tuple_delimiter}舌象: 热象){record_delimiter}
("Entity"{tuple_delimiter}苔黄腻{tuple_delimiter}舌苔{tuple_delimiter}苔黄腻提示痰热内蕴。{tuple_delimiter}舌象: 痰热){record_delimiter}
("Entity"{tuple_delimiter}脉滑数{tuple_delimiter}脉象{tuple_delimiter}脉滑数与痰热壅肺相吻合。{tuple_delimiter}脉象: 痰热, 实热){record_delimiter}
("Entity"{tuple_delimiter}痰热壅肺{tuple_delimiter}病机{tuple_delimiter}痰热壅肺导致肺失宣肃，进而出现咳嗽痰黄等表现。{tuple_delimiter}病位: 肺, 病性: 热, 邪气: 痰){record_delimiter}
("Entity"{tuple_delimiter}清热肃肺{tuple_delimiter}治法{tuple_delimiter}清热肃肺用于清解肺热并恢复肺之宣肃。{tuple_delimiter}治疗目标: 清热, 宣肺){record_delimiter}
("Entity"{tuple_delimiter}清金化痰汤{tuple_delimiter}方剂{tuple_delimiter}清金化痰汤用于痰热壅肺所致的咳嗽痰黄。{tuple_delimiter}用途: 痰热咳嗽){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}痰热壅肺{tuple_delimiter}咳嗽痰黄{tuple_delimiter}痰热壅肺会导致咳嗽痰黄等症状。{tuple_delimiter}病机, 症状{tuple_delimiter}9){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}痰热壅肺{tuple_delimiter}清热肃肺{tuple_delimiter}清热肃肺是针对痰热壅肺的核心治法。{tuple_delimiter}病机, 治法{tuple_delimiter}9){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}清热肃肺{tuple_delimiter}清金化痰汤{tuple_delimiter}清金化痰汤可体现清热肃肺、化痰止咳的治疗思路。{tuple_delimiter}治法, 方剂{tuple_delimiter}8){record_delimiter}
("High-order Hyperedge"{tuple_delimiter}痰热壅肺{tuple_delimiter}咳嗽痰黄{tuple_delimiter}舌红{tuple_delimiter}苔黄腻{tuple_delimiter}脉滑数{tuple_delimiter}痰热壅肺可由咳嗽痰黄、舌红、苔黄腻、脉滑数等共同支持，是一个完整的证候判别集合。{tuple_delimiter}痰热壅肺证候判别{tuple_delimiter}证候, 舌脉, 症状{tuple_delimiter}9){completion_delimiter}
#############################""",
    """Example 2:

Entity_types: [病名, 证型, 病机, 症状, 治法, 方剂, 药材, 加减条件]
Text:
呕吐酸腐，脘腹胀满，嗳气厌食，舌苔垢腻，脉滑实。病机为食滞内停，胃失和降。治法为消食导滞，和胃降逆。可用保和丸加减。若腹满便秘，加枳实、大黄。
################
Output:
("Entity"{tuple_delimiter}呕吐酸腐{tuple_delimiter}症状{tuple_delimiter}呕吐酸腐是食滞内停的重要表现。{tuple_delimiter}病位: 胃){record_delimiter}
("Entity"{tuple_delimiter}食滞内停{tuple_delimiter}病机{tuple_delimiter}食滞内停导致胃失和降而见呕吐酸腐、脘腹胀满。{tuple_delimiter}病位: 胃, 病性: 实){record_delimiter}
("Entity"{tuple_delimiter}消食导滞{tuple_delimiter}治法{tuple_delimiter}消食导滞用于解除食滞，恢复胃气和降。{tuple_delimiter}治疗目标: 导滞和胃){record_delimiter}
("Entity"{tuple_delimiter}保和丸{tuple_delimiter}方剂{tuple_delimiter}保和丸是食滞内停型呕吐的常用方。{tuple_delimiter}用途: 食滞呕吐){record_delimiter}
("Entity"{tuple_delimiter}腹满便秘{tuple_delimiter}加减条件{tuple_delimiter}腹满便秘是需要加强导滞通腑的条件。{tuple_delimiter}条件: 腑气不通){record_delimiter}
("Entity"{tuple_delimiter}枳实{tuple_delimiter}药材{tuple_delimiter}枳实可用于导滞消痞。{tuple_delimiter}作用: 行气导滞){record_delimiter}
("Entity"{tuple_delimiter}大黄{tuple_delimiter}药材{tuple_delimiter}大黄可通腑泄热，帮助导滞。{tuple_delimiter}作用: 通腑){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}食滞内停{tuple_delimiter}保和丸{tuple_delimiter}保和丸适用于食滞内停所致的呕吐。{tuple_delimiter}病机, 方剂{tuple_delimiter}9){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}腹满便秘{tuple_delimiter}大黄{tuple_delimiter}腹满便秘时可加大黄以导滞通腑。{tuple_delimiter}条件, 加减{tuple_delimiter}8){record_delimiter}
("High-order Hyperedge"{tuple_delimiter}食滞内停{tuple_delimiter}消食导滞{tuple_delimiter}保和丸{tuple_delimiter}枳实{tuple_delimiter}大黄{tuple_delimiter}该集合体现了食滞内停型呕吐从病机到治法、方剂及加减药材的完整治疗路径。{tuple_delimiter}食滞呕吐治疗路径{tuple_delimiter}病机, 治法, 方药, 加减{tuple_delimiter}9){completion_delimiter}
#############################""",
    """Example 3:

Entity_types: [药材, 功效, 禁忌, 注意事项, 特殊人群]
Text:
三棱破血行气，消积止痛。孕妇慎用。月经过多者不宜久服。
################
Output:
("Entity"{tuple_delimiter}三棱{tuple_delimiter}药材{tuple_delimiter}三棱为活血消积类药材。{tuple_delimiter}功效: 破血行气, 消积止痛){record_delimiter}
("Entity"{tuple_delimiter}孕妇慎用{tuple_delimiter}禁忌{tuple_delimiter}孕妇使用三棱需谨慎。{tuple_delimiter}特殊人群: 妊娠){record_delimiter}
("Entity"{tuple_delimiter}月经过多者{tuple_delimiter}特殊人群{tuple_delimiter}月经过多者服用三棱存在风险。{tuple_delimiter}风险: 出血增多){record_delimiter}
("Entity"{tuple_delimiter}不宜久服{tuple_delimiter}注意事项{tuple_delimiter}三棱不宜长期连续服用。{tuple_delimiter}用药时长: 需控制){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}三棱{tuple_delimiter}孕妇慎用{tuple_delimiter}三棱对孕妇存在用药风险，因此需慎用。{tuple_delimiter}药材, 禁忌{tuple_delimiter}9){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}三棱{tuple_delimiter}不宜久服{tuple_delimiter}三棱活血之力较强，不宜久服。{tuple_delimiter}药材, 注意事项{tuple_delimiter}8){record_delimiter}
("High-order Hyperedge"{tuple_delimiter}三棱{tuple_delimiter}孕妇慎用{tuple_delimiter}月经过多者{tuple_delimiter}不宜久服{tuple_delimiter}该集合体现了三棱在特殊人群和用药时长上的安全约束。{tuple_delimiter}三棱安全约束{tuple_delimiter}药材, 禁忌, 特殊人群, 注意事项{tuple_delimiter}9){completion_delimiter}
#############################""",
    """Example 4:

Entity_types: [病名, 证型, 病机, 治法, 方剂, 生理时期, 特殊人群, 注意事项]
Text:
经行腹痛，经血量少，色暗有块，痛甚于经前，舌暗，脉弦涩。辨为气滞血瘀证。治法为理气活血，化瘀止痛。可参考膈下逐瘀汤加减。月经期应注意保暖，避免生冷。
################
Output:
("Entity"{tuple_delimiter}经行腹痛{tuple_delimiter}病名{tuple_delimiter}经行腹痛是妇科常见病证。{tuple_delimiter}人群: 女性, 时相: 经期){record_delimiter}
("Entity"{tuple_delimiter}气滞血瘀证{tuple_delimiter}证型{tuple_delimiter}气滞血瘀证可见经前腹痛、经血暗有块等表现。{tuple_delimiter}病性: 瘀, 郁滞: 气血){record_delimiter}
("Entity"{tuple_delimiter}理气活血{tuple_delimiter}治法{tuple_delimiter}理气活血用于疏解气滞并化瘀止痛。{tuple_delimiter}治疗目标: 行气化瘀){record_delimiter}
("Entity"{tuple_delimiter}膈下逐瘀汤{tuple_delimiter}方剂{tuple_delimiter}膈下逐瘀汤可用于气滞血瘀型经行腹痛。{tuple_delimiter}用途: 妇科瘀血疼痛){record_delimiter}
("Entity"{tuple_delimiter}月经期{tuple_delimiter}生理时期{tuple_delimiter}月经期是本病证判断和调护的重要时相。{tuple_delimiter}时相: 经期){record_delimiter}
("Entity"{tuple_delimiter}避免生冷{tuple_delimiter}注意事项{tuple_delimiter}经期应避免生冷，以免加重气血运行不畅。{tuple_delimiter}调护: 经期保养){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}气滞血瘀证{tuple_delimiter}理气活血{tuple_delimiter}理气活血是针对气滞血瘀证的核心治法。{tuple_delimiter}证型, 治法{tuple_delimiter}9){record_delimiter}
("Low-order Hyperedge"{tuple_delimiter}理气活血{tuple_delimiter}膈下逐瘀汤{tuple_delimiter}膈下逐瘀汤能够体现理气活血、化瘀止痛的治疗思路。{tuple_delimiter}治法, 方剂{tuple_delimiter}8){record_delimiter}
("High-order Hyperedge"{tuple_delimiter}经行腹痛{tuple_delimiter}气滞血瘀证{tuple_delimiter}理气活血{tuple_delimiter}膈下逐瘀汤{tuple_delimiter}月经期{tuple_delimiter}该集合体现了妇科病在病名、证型、治法、方剂与生理时期上的完整关联。{tuple_delimiter}妇科经期辨治路径{tuple_delimiter}病名, 证型, 治法, 方剂, 时相{tuple_delimiter}9){completion_delimiter}
#############################""",
]


TCM_OVERRIDES: dict[str, Any] = {
    "DEFAULT_LANGUAGE": "Chinese",
    "DEFAULT_ENTITY_TYPES": TCM_DEFAULT_ENTITY_TYPES,
    "theme_extraction": """-Goal-
给定一段中医知识文本，请总结该段的辨证治疗主轴，并识别支撑该主轴的关键键实体。
使用 {language} 作为输出语言。

-Steps-

1. 总结该段文本的核心主轴。这里的 theme 不是文学主题，而是该段的证机-治法-方证或安全约束主线。
请抽取：
- theme_description: 用一句话概括该段最核心的辨证治疗主轴，尽量包含病机、治法、方药方向或适用条件。
格式：
("theme"{tuple_delimiter}<theme_description>)

2. 从该主轴中抽取关键键实体。只保留真正决定检索方向的高价值键，例如病机键、治法键、证型键、方剂键、安全键、人群/时期键。
对每个键实体抽取：
- key_entity_name: 键实体名称，保持原文语言。
- key_entity_type: 键实体类型，例如病机键、治法键、证型键、方剂键、安全键、人群键、症状键。
- key_entity_description: 简明说明该键实体为什么是本段主轴的重要支点。
- key_score: 0 到 100 的重要性分数。
格式：
("key_entity"{tuple_delimiter}<key_entity_name>{tuple_delimiter}<key_entity_type>{tuple_delimiter}<key_entity_description>{tuple_delimiter}<key_score>)

3. 将步骤 1 和步骤 2 的结果作为单一列表输出，列表项之间使用 **{record_delimiter}** 分隔。

4. 结束时输出 {completion_delimiter}

######################
-Examples-
######################
{examples}
######################
-Real Data-
######################
Text: {input_text}
######################
Output:
""",
    "theme_extraction_examples": TCM_THEME_EXAMPLES,
    "entity_extraction": """-Goal-
给定一段中医知识文本和实体类型列表，从文本中识别这些实体，并构建实体之间的低阶与高阶超边关系。
使用 {language} 作为输出语言。

-Steps-

1. 识别文本中的实体。
对每个实体抽取：
- entity_name: 实体名称，保持原文语言。
- entity_type: 尽量从 [{entity_types}] 中选择最合适的类型。
- entity_description: 对实体在本段中的医学含义、作用或表现进行简要说明。
- additional_properties: 其他与该实体直接相关的属性，例如病位、病性、人群、时相、剂量、调护条件等。
格式：
("Entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>{tuple_delimiter}<additional_properties>)

2. 在识别出的实体中，抽取所有明确存在的低阶关系。
对每对相关实体抽取：
- entities_pair
- low_order_relationship_description
- low_order_relationship_keywords
- low_order_relationship_strength
格式：
("Low-order Hyperedge"{tuple_delimiter}<entity_name1>{tuple_delimiter}<entity_name2>{tuple_delimiter}<low_order_relationship_description>{tuple_delimiter}<low_order_relationship_keywords>{tuple_delimiter}<low_order_relationship_strength>)

3. 尽可能识别多个实体共同组成的高阶关联集合，优先围绕以下链路构造：
- 证型 + 症状 + 舌质 + 舌苔 + 脉象 + 病机
- 证型 + 治法 + 方剂 + 药材 + 加减条件
- 方剂/药材 + 禁忌 + 特殊人群 + 生理时期
对每个高阶集合抽取：
- entities_set
- high_order_relationship_description
- high_order_relationship_generalization
- high_order_relationship_keywords
- high_order_relationship_strength
格式：
("High-order Hyperedge"{tuple_delimiter}<entity_name1>{tuple_delimiter}<entity_name2>{tuple_delimiter}<entity_nameN>{tuple_delimiter}<high_order_relationship_description>{tuple_delimiter}<high_order_relationship_generalization>{tuple_delimiter}<high_order_relationship_keywords>{tuple_delimiter}<high_order_relationship_strength>)

4. 将步骤 1 到步骤 3 的结果作为单一列表输出，列表项之间使用 **{record_delimiter}** 分隔。

5. 结束时输出 {completion_delimiter}

######################
-Examples-
######################
{examples}
######################
-Real Data-
######################
Entity_types: [{entity_types}]
Text: {input_text}
######################
Output:
""",
    "entity_extraction_examples": TCM_ENTITY_EXAMPLES,
    "fail_response": "抱歉，基于当前检索到的证据，我还不能可靠回答这个问题。",
    "rag_response": """---Role---

你是一名谨慎的中医知识检索增强助手，只能依据给定数据表中的证据作答。

---Goal---

请根据给定数据表回答用户问题，优先组织成中医辨证逻辑：
- 辨证判断
- 病机说明
- 治法与方药
- 注意事项或适用条件

如果证据不足，请明确说明证据不足。
不要编造，不要把缺失信息补全成确定结论。
如果问题涉及个体化诊疗而表中没有足够证据，请保留不确定性。

---Target response length and format---

{response_type}

---Data tables---

{context_data}

请按需要使用 markdown 组织回答。
""",
    "theme_keywords_extraction": """---Role---

你是一名中医检索助手，负责从用户问题中抽取主题级关键词。

---Goal---

给定用户问题，抽取能代表辨证主线的主题级关键词，优先关注：
- 证机主轴
- 治法主轴
- 方证主轴
- 安全主轴

---Instructions---

- 主题关键词要偏抽象主线，而不是零散细节。
- 如果问题主要在问禁忌、人群或时期，也可以把安全主轴作为主题关键词。
- 仅输出 JSON。
- JSON 只有一个键： "theme_keywords"

######################
-Examples-
######################
Example 1:

Query: "咳嗽痰黄、舌红苔黄腻，这种情况偏什么证，治法是什么？"
################
Output:
{{
  "theme_keywords": ["痰热壅肺", "清热化痰", "咳嗽辨证"]
}}
#############################
Example 2:

Query: "脾胃虚寒型呕吐为什么要温中降逆？"
################
Output:
{{
  "theme_keywords": ["脾胃虚寒", "温中降逆", "呕吐病机"]
}}
#############################
Example 3:

Query: "孕妇能不能用三棱？"
################
Output:
{{
  "theme_keywords": ["孕期用药安全", "活血药禁忌", "特殊人群"]
}}
#############################
-Real Data-
######################
Query: {query}
######################
Output:
""",
    "entity_keywords_extraction": """---Role---

你是一名中医检索助手，负责从用户问题中抽取实体级关键词。

---Goal---

给定用户问题，抽取具体、可检索的实体级关键词，优先关注：
- 病名
- 证型
- 症状
- 舌质
- 舌苔
- 脉象
- 方剂
- 药材
- 人群条件

---Instructions---

- 关键词应尽量具体，贴近原始临床表达。
- 不要只给抽象概念，要保留可落地检索的症状和实体。
- 仅输出 JSON。
- JSON 只有一个键： "entity_keywords"

######################
-Examples-
######################
Example 1:

Query: "咳嗽痰黄、舌红苔黄腻，这种情况偏什么证，治法是什么？"
################
Output:
{{
  "entity_keywords": ["咳嗽", "痰黄", "舌红", "苔黄腻"]
}}
#############################
Example 2:

Query: "食滞内停型呕吐一般用什么方，还要看哪些症状？"
################
Output:
{{
  "entity_keywords": ["食滞内停", "呕吐酸腐", "脘腹胀满", "保和丸"]
}}
#############################
Example 3:

Query: "孕妇能不能用三棱？"
################
Output:
{{
  "entity_keywords": ["三棱", "孕妇"]
}}
#############################
-Real Data-
######################
Query: {query}
######################
Output:
""",
    "naive_rag_response": """你是一名谨慎的中医知识助手。
以下是你已知的知识：
{content_data}
---
如果知识不足以支持回答，请直接说明不知道或证据不足。
不要编造，不要把未出现的信息说成确定事实。
请根据用户问题给出目标长度和格式的回答，优先保持辨证逻辑与证据一致。
---Target response length and format---
{response_type}
""",
    "rag_define": """
根据已有分析，问题中潜在的重要线索包括：
{{ {entity_keywords} | {theme_keywords} }}
请结合这些线索和上下文证据，优先围绕辨证主线组织答案，不要机械拼接知识片段。
""",
    "entity_keywords_aglin": """---Role---

你是一名中医检索助手，负责在给定主题答案后进一步抽取与之对齐的实体级关键词。

---Goal---

给定用户问题和第一阶段主题答案，抽取与该主题主轴最相关的具体实体关键词。

---Instructions---

- 优先抽取与主题主轴一致的病名、症状、舌脉、方剂、药材、人群条件。
- 如果主题答案强调安全约束，就优先抽取药材、禁忌对象和时期条件。
- 仅输出 JSON：{{"entity_keywords": [...]}}

######################
-Examples-
######################
Query: "咳嗽痰黄、舌红苔黄腻，偏什么证？"
Theme: "本题主轴为痰热壅肺，治宜清热化痰。"

Output:
{{
  "entity_keywords": ["咳嗽", "痰黄", "舌红", "苔黄腻", "脉滑数"]
}}
#############################

Query: "孕妇能不能用三棱？"
Theme: "本题主轴为孕期用药安全，需要核对三棱的禁忌与慎用人群。"

Output:
{{
  "entity_keywords": ["三棱", "孕妇", "禁忌", "慎用"]
}}
#############################

-Real Data-
######################
Query: {query}
Theme: {theme}
######################
Output:
""",
    "rag_define_aglin": """
根据已有分析，当前主题答案与实体级关键词为：
{{ {theme_answer} | {entity_keywords} }}
请结合主题答案和实体关键词组织最终回答，优先保证辨证链路与证据引用一致。
""",
}


def build_tcm_prompts() -> dict[str, Any]:
    prompts = deepcopy(_DEFAULT_PROMPTS_SNAPSHOT)
    prompts.update(deepcopy(TCM_OVERRIDES))
    return prompts


def apply_tcm_prompts() -> dict[str, Any]:
    ACTIVE_PROMPTS.clear()
    ACTIVE_PROMPTS.update(build_tcm_prompts())
    return ACTIVE_PROMPTS


def restore_default_prompts() -> dict[str, Any]:
    ACTIVE_PROMPTS.clear()
    ACTIVE_PROMPTS.update(deepcopy(_DEFAULT_PROMPTS_SNAPSHOT))
    return ACTIVE_PROMPTS


__all__ = [
    "TCM_DEFAULT_ENTITY_TYPES",
    "build_tcm_prompts",
    "apply_tcm_prompts",
    "restore_default_prompts",
]
