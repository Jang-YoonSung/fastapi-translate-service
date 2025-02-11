# HF token 설정
from huggingface_hub import HfFolder
import torch
import time

from transformers import BitsAndBytesConfig

quantization_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)

from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "google/madlad400-3b-mt"
model = T5ForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="cuda:0"
    )
tokenizer = T5Tokenizer.from_pretrained(model_name)

madlad_abt = load_dataset("allenai/madlad-400", "abt")

# Load model directly
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained(
#     "facebook/nllb-200-3.3B",
#     device_map="cuda:1",
# )
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")

# tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
# model = AutoModelForSeq2SeqLM.from_pretrained(
#     "facebook/nllb-200-distilled-1.3B",
#     quantization_config=quantization_config,
#     device_map="cuda:0",
#     )
# tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")


# text = "### 한국어: 한국에서 가장 높은 건물은 롯데타워 입니다.</끝>\n\
# ### 일본어: "
# text = "### 일본어: 韓国で最も高い建物はロッテタワーです。</끝>\n\
# ### 한국어: "




# texts = """We introduce MADLAD-400, a manually audited, general domain 3T token monolingual dataset based on CommonCrawl, spanning 419 languages.
# We discuss the limitations revealed by self-auditing MADLAD-400, and the role data auditing had in the dataset creation process.
# We then train and release a 10.7B-parameter multilingual machine translation model on 250 billion tokens covering over 450 languages using publicly available data, and find that it is competitive with models that are significantly larger, and report the results on different domains.
# In addition, we train a 8B-parameter language model, and assess the results on few-shot translation. We make the baseline models 1 available to the research community.""" 





# texts = """グリーンランドで北極を調査していた米国航空宇宙局(NASA)の科学者たちが氷上の下で「秘密基地」を発見しました。
# 26日（以下現地時間）、アメリカの科学専門メディアスペースドットコムは「NASAの科学者たちがグリーンランドの氷上を飛びながら研究のためのデータを収集していたところ、冷戦時代に作られた米軍基地である『キャンプセンチュリー』を発見した」と報道しました。
# 4月、NASAの研究陣はレーダー装備を搭載した航空機に乗ってグリーンランド上空を飛行しました。 レーダーを通じてグリーンランド氷床の深さと氷床下の岩盤層を分析し、これを地図で表示する研究のためのデータ収集過程でした。
# 研究陣はレーダー装備を見ながら氷上を飛び、レーダーが「ある物体」を感知しては点滅することを確認した。 その後、レーダーが収集した情報を基に氷上の下を地図で具現すると、驚くべきことに色々なトンネルで構成された「氷都市」が現れました。
# グリーンランドの氷上の下で捉えられた氷の都市の正体は「キャンプ·センチュリー」で、1959年に建設されたアメリカ軍の秘密基地です。
# キャンプセンチュリーは約4000キロの膨大なトンネルで構成されており、最低気温氷点下57度·最高風速時速197キロ/hに達する極限の環境に作られました。
# 冷戦時代、米国は旧ソ連を狙ってミサイル戦略基地としてキャンプセンチュリーを建設しましたが、維持費用が多くかかるうえ、氷河の不安定性のためにトンネルが崩壊する危険性が高まり、結局1967年に閉鎖しました。
# 建設当初は地表面ともっと近かったが、約60年間捨てられた状態が維持され、その上に30㎝ほどの雪と氷が積もったと知られています。
# 現在、氷上の下に眠っているキャンプセンチュリーには放射性廃棄物を含む危険物質がそのまま埋められており、環境問題が憂慮されています。
# NASAジェット推進研究所(JPL)の氷河専門家であるチャド·グリーン博士は公式声明で「グリーンランドの氷層を調査していたところ、何かがレーダーに捉えられた。 私たちは最初にその物体が何なのか全く知らなかった」とし「収集されたデータによればキャンプセンチュリーの各々の構造物は以前に知っていたことと異なる部分があった」と明らかにした。
# 続けて「地球温暖化で海の水温と大気の温度が急速に上昇する環境で氷上がどのように変化するか分からない」とし「キャンプセンチュリーは地球の気候変化がグリーンランドのような地域にどんな影響を及ぼすのかあらかじめ予測できる道標の役割をする」と説明した。
# 一方、1950年代後半、米軍はキャンプ·センチュリーを「氷の下の都市」と呼び、科学基地であることを強調しましたが、実は「プロジェクト·アイスワーム」（Iceworm）という名の軍事プロジェクトの一環でした。 トンネル21個を開けて旧ソ連の目の前に核ミサイル600基を隠すことが目的でした。
# しかし、氷上が予想したより早く動き、トンネルの形がねじれて雪の重さで崩壊の危険まで提起され、氷の下の秘密核基地建設は失敗に終わった。"""


# for text in texts.split('\n'):
#     start = time.time()
#     inputs = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
#     # inputs = model.generate(inputs, forced_bos_token_id=tokenizer.lang_code_to_id['eng_Latn'], max_length=512)
#     outputs = model.generate(inputs, forced_bos_token_id=tokenizer.lang_code_to_id['ces_Latn'], max_length=512) # kor_Hang 한국어, eng_Latn 영어, jpn_Jpan 일본어, zho_Hans 중국어, ces_Latn 체코어
#     end = time.time()
#     elapsed_time = end - start
#     print(tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
#     # print(f"번역 소요 시간: {elapsed_time:.2f}초")

# 일본어 번역 태그 <2ja>, 한국어 번역 태그 <2ko>, 영어 번역 태그 <2en>, 체코어 번역 태그 <2cs>, 중국어 번역 태그 <2zh>
# for text in texts.split('\n'):
#     start = time.time()
#     input_ids = tokenizer("<2ko> " + text, return_tensors="pt").input_ids.to(model.device)
#     outputs = model.generate(input_ids=input_ids, max_length=512)
#     end = time.time()
#     elapsed_time = end - start
#     print(tokenizer.decode(outputs[0], skip_special_tokens=True))
#     # print(f"번역 소요 시간: {elapsed_time:.2f}초")

# tokenizer.decode(outputs[0], skip_special_tokens=True)



# from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained(
#     # "Bllossom/llama-3.2-Korean-Bllossom-3B",
#     "meta-llama/Llama-3.2-3B-Instruct",
#     # quantization_config = quantization_config,
#     device_map = "cuda:0"
# )

# # Tokenizer 설정
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# message = [
#     {"role":"user", "content":"안녕하세요 나는 코딩을 공부하는 학생입니다. 를 중국어로 번역해줘"}
#     ]

# input_ids = tokenizer.apply_chat_template(
#     message,
#     add_generation_prompt=True,
#     return_tensors="pt"
# ).to(model.device)

# terminators = [
#     tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
#     tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]

# outputs = model.generate(
#     input_ids,
#     max_new_tokens=256,
#     eos_token_id=terminators,
#     use_cache = True,
#     do_sample=True,
#     temperature=0.05,
#     top_p=0.95,
# )

# print(tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True))