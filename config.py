# -*- coding: utf-8 -*-
"""
户晨风对话机器人 - 配置文件
"""
import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据目录 - 存放户晨风文字稿的目录
DATA_DIR = os.path.join(BASE_DIR, '..', 'HuChenFeng-main')

# 机器人名称
BOT_NAME = "户子"

# 户晨风在对话中的标识
HOST_SPEAKER = "户晨风"

# 网友在对话中的标识  
CALLER_SPEAKERS = ["某网友", "网友"]

# 对话最小字数过滤（太短的对话没有训练价值）
MIN_RESPONSE_LENGTH = 10

# 最大上下文窗口（用于提取多轮对话）
MAX_CONTEXT_WINDOW = 3

# TF-IDF 参数
TFIDF_MAX_FEATURES = 50000
TFIDF_TOP_N = 5  # 返回最相似的N个回复进行候选

# 相似度阈值
SIMILARITY_THRESHOLD = 0.15

# 关键词权重提升
KEYWORD_BOOST = 1.5

# 是否启用上下文感知
ENABLE_CONTEXT = True

# 日志级别
LOG_LEVEL = "INFO"

# 最大语料库大小（限制索引规模，加快检索）
MAX_CORPUS_SIZE = 30000

# 缓存目录
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
