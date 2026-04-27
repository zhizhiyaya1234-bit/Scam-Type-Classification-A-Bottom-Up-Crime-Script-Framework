import pandas as pd
import numpy as np
import re
from pathlib import Path


# 1. 读取原始数据

input_path = "/Users/wangziyi/Desktop/news_202603182039.csv"
df = pd.read_csv(input_path)

print("原始数据量：", len(df))
print("原始字段：", df.columns.tolist())


# 2. 只保留后续需要的核心字段

keep_cols = [
    "news_id", "title", "content", "summary", "publish_date",
    "source_publication", "country", "language", "translated_content",
    "topic_category", "url", "source", "updated_time", "status"
]

existing_cols = [c for c in keep_cols if c in df.columns]
df = df[existing_cols].copy()


# 3. 基础文本清洗函数
def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x)

    x = x.replace("\u00a0", " ")
    x = x.replace("\t", " ")
    x = x.replace("\r", " ")
    x = re.sub(r"\n+", "\n", x)
    x = re.sub(r"[ ]+", " ", x)

    # 去掉常见尾注
    x = re.sub(r"For Reprint Rights:.*", "", x, flags=re.IGNORECASE)
    x = re.sub(r"Word count:\s*\d+", "", x, flags=re.IGNORECASE)

    # 去掉网址
    x = re.sub(r"https?://\S+", "", x)

    x = x.strip()
    return x


for col in ["title", "content", "summary", "translated_content"]:
    if col in df.columns:
        df[col] = df[col].apply(clean_text)


# 4. 合并文本字段

def merge_text(row):
    parts = []
    for col in ["title", "content", "translated_content", "summary"]:
        if col in row.index:
            value = row[col]
            if pd.notna(value) and str(value).strip():
                parts.append(str(value).strip())

    merged = "\n".join(parts)
    merged = re.sub(r"\n{2,}", "\n", merged).strip()
    return merged

df["combined_text"] = df.apply(merge_text, axis=1)
df["combined_len"] = df["combined_text"].str.len()

# 5. 删除空白和极短文本

before = len(df)
df = df[df["combined_text"].str.strip() != ""].copy()
print("删除空白文本：", before - len(df))

before = len(df)
df = df[df["combined_len"] >= 80].copy()
print("删除过短文本(<80字符)：", before - len(df))

# 6. 时间清洗与筛选
# 时间范围：2025-01-01 到 2025-12-31
# 优先 publish_date，其次 updated_time
def parse_datetime_to_naive(series):
    """
    统一解析为 pandas datetime：
    1. 先强制按 UTC 解析
    2. 再去掉时区，变成 naive datetime
    """
    s = series.fillna("").astype(str).str.strip()
    s = s.str.replace(".", "-", regex=False)
    s = s.str.replace("/", "-", regex=False)

    dt = pd.to_datetime(s, errors="coerce", utc=True)
    dt = dt.dt.tz_convert(None)   # 去掉 UTC 时区，变成普通 datetime64[ns]
    return dt


date_source = None

if "publish_date" in df.columns:
    df["publish_dt"] = parse_datetime_to_naive(df["publish_date"])
    if df["publish_dt"].notna().sum() > 0:
        date_source = "publish_dt"

if date_source is None and "updated_time" in df.columns:
    df["updated_dt"] = parse_datetime_to_naive(df["updated_time"])
    if df["updated_dt"].notna().sum() > 0:
        date_source = "updated_dt"

if date_source is None:
    raise ValueError("没有找到可解析的日期列（publish_date / updated_time）")

print("使用日期字段：", date_source)
print("日期列类型：", df[date_source].dtype)
print("可解析日期条数：", df[date_source].notna().sum())

start_date = pd.Timestamp("2025-01-01")
end_date = pd.Timestamp("2025-12-31 23:59:59")

before = len(df)
df = df[df[date_source].notna()].copy()
print("删除无法解析日期的记录：", before - len(df))

before = len(df)
df = df[(df[date_source] >= start_date) & (df[date_source] <= end_date)].copy()
print("删除不在 2025 年范围内的记录：", before - len(df))

df["filter_date"] = df[date_source]
print("剩余有效数据：", len(df))

# 7. 状态清洗：只保留 active

if "status" in df.columns:
    before = len(df)
    df["status"] = df["status"].fillna("").astype(str).str.strip().str.lower()
    df = df[(df["status"] == "active") | (df["status"] == "")].copy()
    print("删除非 active 状态：", before - len(df))

# 8. 去重

def normalize_for_dedup(x):
    x = str(x).lower()
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"[^\w\u4e00-\u9fff ]", "", x)
    x = x.strip()
    return x

if "title" in df.columns:
    df["title_norm"] = df["title"].fillna("").apply(normalize_for_dedup)
else:
    df["title_norm"] = ""

df["text_norm"] = df["combined_text"].fillna("").apply(normalize_for_dedup)

# 8.1 按 url 去重
if "url" in df.columns:
    before = len(df)
    df["url"] = df["url"].fillna("").astype(str).str.strip()
    df = df.drop_duplicates(subset=["url"], keep="first")
    print("按 url 去重：", before - len(df))

# 8.2 按 news_id 去重
if "news_id" in df.columns:
    before = len(df)
    df["news_id"] = df["news_id"].fillna("").astype(str).str.strip()
    df = df.drop_duplicates(subset=["news_id"], keep="first")
    print("按 news_id 去重：", before - len(df))

# 8.3 按标准化全文去重
before = len(df)
df = df.drop_duplicates(subset=["text_norm"], keep="first")
print("按标准化全文去重：", before - len(df))

# 8.4 按标题+日期再辅助去重
before = len(df)
df["filter_date_str"] = df["filter_date"].dt.strftime("%Y-%m-%d")
df = df.drop_duplicates(subset=["title_norm", "filter_date_str"], keep="first")
print("按 标题+日期 去重：", before - len(df))

# 9. 标记语言与翻译情况

if "language" in df.columns:
    df["language"] = df["language"].fillna("").astype(str).str.strip().str.lower()

if "translated_content" in df.columns:
    df["has_translated_content"] = df["translated_content"].fillna("").astype(str).str.strip().ne("")
else:
    df["has_translated_content"] = False

# 10. 生成适合 LLM 的输入字段

def build_llm_input(row):
    title = row["title"] if "title" in row.index else ""
    content = row["content"] if "content" in row.index else ""
    translated = row["translated_content"] if "translated_content" in row.index else ""

    return f"标题：{title}\n正文：{content}\n翻译正文：{translated}".strip()

df["llm_input"] = df.apply(build_llm_input, axis=1)

# 11. 添加本地索引

df = df.reset_index(drop=True)
df["local_id"] = ["news_2025_" + str(i).zfill(6) for i in range(len(df))]

# 12. 输出文件

output_dir = Path(".")
full_output = output_dir / "news_cleaned_2025_full.csv"
llm_output = output_dir / "news_cleaned_2025_for_llm.csv"
sample_output = output_dir / "news_2025_llm_test_sample.csv"

# 完整清洗版
df.to_csv(full_output, index=False, encoding="utf-8-sig")

# 适合 LLM 的版本
llm_cols = [
    c for c in [
        "local_id", "news_id", "filter_date", "publish_date", "updated_time",
        "source_publication", "country", "language", "url",
        "title", "content", "translated_content", "combined_text", "llm_input"
    ] if c in df.columns
]
df[llm_cols].to_csv(llm_output, index=False, encoding="utf-8-sig")

# 抽样 100 条测试
sample_df = df[llm_cols].sample(min(100, len(df)), random_state=42)
sample_df.to_csv(sample_output, index=False, encoding="utf-8-sig")

print("\n清洗完成")
print("清洗后数据量：", len(df))
print("输出文件：")
print("1.", full_output)
print("2.", llm_output)
print("3.", sample_output)