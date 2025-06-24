from enum import Enum
from pydantic import BaseModel, Field
from typing import List
import json
from string import Template

# Few-shot examples
samples = [
    {
        "input": (
            "慢性関節リウマチの診断と管理\n\n"
            + "関節の腫脹、疼痛、可動域制限などの症状が認められる。"
            + "血清リウマトイド因子やCRPの上昇が典型的な検査所見として報告されている。"
            + "進行例では骨侵食や関節変形を合併し、日常生活動作に支障をきたすことがある。"
                  ),
        "output": {
            "results": [
                {
                "disease_text": "慢性関節リウマチ",
                "abnormal_findings_caused_by_the_disease": [
                    {
                    "finding_text": "関節の腫脹",
                    "finding_type": "symptom"
                    },
                    {
                    "finding_text": "疼痛",
                    "finding_type": "symptom"
                    },
                    {
                    "finding_text": "可動域制限",
                    "finding_type": "symptom"
                    },
                    {
                    "finding_text": "血清リウマトイド因子の上昇",
                    "finding_type": "examination_result"
                    },
                    {
                    "finding_text": "CRPの上昇",
                    "finding_type": "examination_result"
                    },
                    {
                    "finding_text": "骨侵食",
                    "finding_type": "complication"
                    },
                    {
                    "finding_text": "関節変形",
                    "finding_type": "complication"
                    }
                ]
                }
            ]
        }
    },
    {
        "input": (
            "糖尿病の病態\n\n"
            + "多尿、口渇、体重減少が一般的な症状として現れる。"
            + "血糖値の上昇やHbA1cの増加が認められる。"
            + "長期的には網膜症や神経障害を合併する。"
                  ),
        "output": {
            "results": [
                {
                "disease_text": "糖尿病",
                "abnormal_findings_caused_by_the_disease": [
                    {
                    "finding_text": "多尿",
                    "finding_type": "symptom"
                    },
                    {
                    "finding_text": "口渇",
                    "finding_type": "symptom"
                    },
                    {
                    "finding_text": "体重減少",
                    "finding_type": "symptom"
                    },
                    {
                    "finding_text": "血糖値の上昇",
                    "finding_type": "examination_result"
                    },
                    {
                    "finding_text": "HbA1cの増加",
                    "finding_type": "examination_result"
                    },
                    {
                    "finding_text": "網膜症",
                    "finding_type": "complication"
                    },
                    {
                    "finding_text": "神経障害",
                    "finding_type": "complication"
                    }
                ]
                }
            ]
            }
    }
]

class FindingCategory(str, Enum):
    symptom = "symptom"
    examination_result = "examination_result"
    complication = "complication"

class AbnormalFinding(BaseModel):
    finding_text: str = Field(
        ...,
        title="所見テキスト",
        description="記事中に記載された症状・検査所見・合併症の名称"
    )
    finding_type: FindingCategory = Field(
        ...,
        title="所見タイプ",
        description="`symptom`／`examination_result`／`complication` のいずれか"
    )

class DiseaseProperty(BaseModel):
    disease_text: str = Field(
        ...,
        title="疾患テキスト",
        description="抽出された疾患名の正式名称"
    )
    abnormal_findings_caused_by_the_disease: List[AbnormalFinding] = Field(
        ...,
        title="異常所見一覧",
        description="当該疾患に関連する全所見"
    )

class ExtractionResult(BaseModel):
    results: List[DiseaseProperty] = Field(
        ...,
        title="抽出結果リスト",
        description="複数疾患対応の抽出結果配列"
    )

    class Config:
        # by_alias=True で alias 名を出力キーにできる
        populate_by_name = True
        json_schema_extra = {
            "example": samples[0]["output"]
        }

prompts = {
    "system": (
        "あなたは日本語医学文献から疾患と関連所見を抽出する専門エンジンです。"
        "以下の JSON Schema に**完全準拠**し、他のテキストを一切含めず JSON のみで出力してください：\n"
        + json.dumps(ExtractionResult.model_json_schema(), ensure_ascii=False)
    ),
    "user": Template((
        " 以下の<<<INPUT>>>と<<<END>>>で囲まれたテキストを処理し、システムプロンプトで定義したJSON Schemaに"
        "**完全に従って**JSONのみを返してください。\n"
        "処理の内容は、「記事中に記載された個々の症状・検査所見・合併症の表現をそのまま抽出して記載すること」です。\n"
        "与えられたテキストに出現する『疾患テキスト』と『所見テキスト』を抽出して、そのまま記載してください。\n"
        "【所見タイプの定義】\n"
        "symptom: 自覚症状・他覚所見に加えて、医師が視診・触診・聴診などの手や簡単な器具を使って確認できる理学所見や体温・脈拍・呼吸数・血圧などのバイタルサイン、尿量や身長・体重など、簡便に確認可能な所見\n"
        "examination_result: 血液検査、X線、超音波、病理検査など、検査オーダーが必要な客観的所見\n"
        "complication: その疾患の進行・長期化で発生しうる合併症のこと。\n"
        "### タスクの対象:\n"
        "<<<INPUT>>>\n"
        "${ArticleText}\n"
        "<<<END>>>\n\n"
        "### Example 1:\n"
        "# 入力\n"
        "${example1_input}\n"
        "# 出力\n"
        "${example1_output}\n\n"
        "### Example 2:\n"
        "# 入力\n"
        "${example2_input}\n"
        "# 出力\n"
        "${example2_output}\n\n"
    ))
}

prompts_vanilla = {
    "system": (
        "あなたは日本語医学文献から疾患と関連所見を抽出する専門エンジンです。"
        "以下の JSON Schema に**完全準拠**し、他のテキストを一切含めず JSON のみで出力してください：\n"
        + json.dumps(ExtractionResult.model_json_schema(), ensure_ascii=False)
    ),
    "user": Template((
        " 以下の<<<INPUT>>>と<<<END>>>で囲まれたテキストを処理し、システムプロンプトで定義したJSON Schemaに"
        "**完全に従って**JSONのみを返してください。\n"
        "処理の内容は、「記事中に記載された個々の症状・検査所見・合併症の表現をそのまま抽出して記載すること」です。\n"
        "与えられたテキストに出現する『疾患テキスト』と『所見テキスト』を抽出して、そのまま記載してください。\n"
        "【所見タイプの定義】\n"
        "symptom: 自覚症状・他覚所見に加えて、医師が視診・触診・聴診などの手や簡単な器具を使って確認できる理学所見や体温・脈拍・呼吸数・血圧などのバイタルサイン、尿量や身長・体重など、簡便に確認可能な所見\n"
        "examination_result: 血液検査、X線、超音波、病理検査など、検査オーダーが必要な客観的所見\n"
        "complication: その疾患の進行・長期化で発生しうる合併症のこと。\n"
        "### タスクの対象:\n"
        "<<<INPUT>>>\n"
        "${ArticleText}\n"
        "<<<END>>>\n\n"
    ))
}
