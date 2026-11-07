# -*- coding: utf-8 -*-
"""
户子对话机器人 - 核心引擎 v3
机器人 = 户晨风本人，以第一人称回答用户问题
优先使用知识库回答，TF-IDF检索作为补充
"""
import re
import random
import logging
import json
import os
from typing import List, Dict, Tuple, Optional

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    BOT_NAME, HOST_SPEAKER, SIMILARITY_THRESHOLD,
    TFIDF_MAX_FEATURES, TFIDF_TOP_N, KEYWORD_BOOST,
    CACHE_DIR, MIN_RESPONSE_LENGTH, MAX_CORPUS_SIZE
)
from data_parser import ConversationParser, get_topic_keywords
from knowledge_base import KnowledgeBase
from hu_knowledge import HuKnowledge

logger = logging.getLogger(__name__)


class HuZiChatbot:
    """
    户子对话机器人
    机器人就是户晨风本人，以第一人称、用自己的观点和语气回答问题
    """

    def __init__(self, use_cache=True):
        self.parser = ConversationParser()
        self.kb = KnowledgeBase()
        self.hu_knowledge = HuKnowledge()
        self.conversation_history = []

        # TF-IDF 检索器（用于精确匹配相似问题）
        self.vectorizer = None
        self.tfidf_matrix = None
        self.search_corpus = []

        # 加载数据
        if use_cache and os.path.exists(os.path.join(CACHE_DIR, 'hu_responses.json')):
            self.parser.load_corpus()
        else:
            logger.info("首次运行，正在解析数据文件...")
            self.parser.parse_all_files()
            self.parser.save_corpus()

        # 构建知识库
        self.kb.build(self.parser.conversations, self.parser.qa_pairs)

        # 构建检索索引
        self._build_search_index()

        logger.info("户子机器人初始化完成！")

    def _build_search_index(self):
        """构建用于检索的 TF-IDF 索引 - 只索引户晨风自己的话"""
        logger.info("正在构建检索索引...")

        self.search_corpus = []
        seen = set()

        # 1. 问答对：用网友的问题做索引，户晨风的回答做回复
        for qa in self.parser.qa_pairs:
            q = qa['question']
            a = qa['answer']
            q_key = q[:80]
            if q_key not in seen and len(q) >= 5 and len(a) >= 30:
                seen.add(q_key)
                self.search_corpus.append({
                    'query': q,
                    'response': a,
                    'source': 'qa',
                })
            if len(self.search_corpus) >= MAX_CORPUS_SIZE:
                break

        # 2. 户晨风的独立回复
        if len(self.search_corpus) < MAX_CORPUS_SIZE:
            for resp in self.parser.hu_responses:
                text = resp['text']
                t_key = text[:80]
                if t_key not in seen and len(text) >= 30:
                    seen.add(t_key)
                    self.search_corpus.append({
                        'query': text,
                        'response': text,
                        'source': 'response',
                    })
                if len(self.search_corpus) >= MAX_CORPUS_SIZE:
                    break

        logger.info("检索语料: {} 条".format(len(self.search_corpus)))

        if not self.search_corpus:
            return

        # 分词 + TF-IDF
        queries = [item['query'] for item in self.search_corpus]
        tokenized = [' '.join(jieba.cut(q)) for q in queries]

        self.vectorizer = TfidfVectorizer(
            max_features=min(TFIDF_MAX_FEATURES, len(tokenized) * 10),
            token_pattern=r'(?u)\b\w+\b',
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(tokenized)
        logger.info("TF-IDF 索引构建完成: {} x {}".format(
            self.tfidf_matrix.shape[0], self.tfidf_matrix.shape[1]))

    def chat(self, user_input):
        """
        核心对话接口 - 以户晨风的身份回答
        """
        user_input = user_input.strip()
        if not user_input:
            return random.choice([
                '嗯？你说啥？',
                '啊？没听清，再说一遍？',
                '你倒是说话呀，愣着干嘛。',
            ])

        # 记录历史
        self.conversation_history.append({
            'role': 'user',
            'text': user_input,
        })

        # 1. 检查是否是打招呼/闲聊
        greeting = self._handle_greeting(user_input)
        if greeting:
            self._record_bot(greeting)
            return greeting

        # 2. 检查是否在问户晨风的个人信息
        personal = self._handle_personal_question(user_input)
        if personal:
            self._record_bot(personal)
            return personal

        # 3. 检测话题
        topics = self._detect_topics(user_input)

        # 4. 生成回答（新流程：知识库优先）
        answer = self._generate_answer(user_input, topics)

        # 5. 记录历史
        self._record_bot(answer)
        return answer

    def _generate_answer(self, user_input, topics):
        """
        生成户晨风式回答 - 新流程

        策略（按优先级）：
        1. 知识库精确匹配（HuKnowledge）
        2. TF-IDF 高置信度匹配
        3. 知识库模糊匹配
        4. TF-IDF 中等匹配
        5. 话题风格化回答
        6. 通用兜底
        """
        # === 1. 知识库精确匹配（最优先！）===
        kb_result = self.hu_knowledge.query(user_input, topics)
        if kb_result and kb_result['score'] >= 5.0:
            response = kb_result['response']
            return self._ensure_first_person(response)

        # === 2. TF-IDF 高置信度匹配 ===
        tfidf_results = self._tfidf_search(user_input, top_k=10)

        if tfidf_results and tfidf_results[0]['score'] > 0.45:
            best = tfidf_results[0]
            response = self._ensure_first_person(best['response'])
            return self._trim_response(response)

        # === 3. 知识库模糊匹配（降低阈值）===
        if kb_result and kb_result['score'] >= 3.0:
            response = kb_result['response']
            return self._ensure_first_person(response)

        # === 4. TF-IDF 话题相关匹配 ===
        if topics:
            topic_result = self._select_topic_result(user_input, topics, tfidf_results)
            if topic_result:
                return topic_result

        # === 5. TF-IDF 中等匹配 ===
        if tfidf_results and tfidf_results[0]['score'] > 0.25:
            best = tfidf_results[0]
            response = self._ensure_first_person(best['response'])
            return self._trim_response(response)

        # === 6. 知识库兜底（极低阈值）===
        if kb_result and kb_result['score'] >= 1.5:
            response = kb_result['response']
            return self._ensure_first_person(response)

        # === 7. 话题风格化回答 ===
        if topics:
            return self._topic_response(user_input, topics)

        # === 8. 通用兜底 ===
        return self._generic_fallback(user_input)

    def _handle_greeting(self, text):
        """处理打招呼 - 只在纯问候时触发"""
        if len(text) > 15:
            return None

        patterns = [
            (r'^(你好|您好|hi|hello|嗨|hey)$', [
                '你好你好，来了啊，坐。想聊点啥？直接说。',
                '来了来了，你好。说吧，有什么想问的？',
                '你好啊，不着急，慢慢说。',
            ]),
            (r'^(在吗|在不在)$', [
                '在呢，说吧。',
                '在在在，你说。',
            ]),
            (r'^(早上好|早安|下午好|晚上好)$', [
                '你好你好，吃了没？',
                '嗯，好，今天怎么样？',
            ]),
            (r'^(谢谢|感谢|多谢|辛苦了)$', [
                '不用谢，应该的。',
                '没事没事，能帮到你就行。',
                '客气啥。',
            ]),
            (r'^(再见|拜拜|bye|告辞|走了)$', [
                '好，再见，生活愉快。',
                '行，那就这样，拜拜。',
                '好嘞，下次见。',
            ]),
            (r'^(你是谁|你叫什么)$', [
                '我是户晨风啊，还能是谁？你连我都不认识了？',
                '我？我就是个直播的，聊聊天，吹吹牛。',
            ]),
        ]

        for pattern, responses in patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return random.choice(responses)
        return None

    def _handle_personal_question(self, text):
        """处理关于户晨风个人信息的问题"""
        # 必须包含"你"或"户晨风"等指向性词汇
        if not re.search(r'(你|户晨风|户子|你本人)', text):
            return None

        # 排除一般性问题
        if len(text) > 25:
            return None

        patterns = [
            (r'(工资|收入|挣多少|赚多少|一个月.*钱)', [
                '我一个月累死累活搞直播挣两三万，跟人家大主播没法比。我哪有钱请演员呢？真没这个钱。',
                '我这个收入嘛，说实话不多，一年搞个三四十万。跟那些大主播比起来真是没法看。',
            ]),
            (r'(粉丝|关注|多少人|火不火)', [
                '粉丝多少无所谓，多点少点真的无所谓的。确实我这个影响力是远大于我的粉丝数的，远远大于。如果匹配的话，我觉得全网应该有个2000万粉丝差不多。',
                '你问这个干嘛？我就是个小主播，粉丝不多，但是影响还行。',
            ]),
            (r'(在哪|哪里人|住哪|城市)', [
                '我现在在成都。经常去各地拍视频，今天在这明天在那的。',
            ]),
            (r'(以前.*工作|之前.*干啥|做过什么)', [
                '我之前是做私募基金研究的，研究各个行业。后来嘛，就做了现在这个。',
            ]),
            (r'(多大了|年龄|几岁)', [
                '这个就不说了吧，你猜。',
            ]),
            (r'(为什么.*直播|为什么.*做这个|为啥.*当主播)', [
                '说白了就是挣钱嘛。我发现做这个能挣钱，虽然不多，但总比没有强。而且我也喜欢跟大家聊聊天。',
            ]),
        ]

        for pattern, responses in patterns:
            if re.search(pattern, text):
                return random.choice(responses)
        return None

    def _detect_topics(self, text):
        """检测用户输入涉及的话题"""
        topics = []
        topic_kws = get_topic_keywords()
        for topic, keywords in topic_kws.items():
            for kw in keywords:
                if kw in text:
                    topics.append(topic)
                    break
        return topics

    def _select_topic_result(self, user_input, topics, tfidf_results):
        """从 TF-IDF 结果中选择与话题匹配的回答"""
        if not tfidf_results:
            return None

        topic_kws = get_topic_keywords()
        topic_keywords_set = set()
        for t in topics:
            for kw in topic_kws.get(t, []):
                topic_keywords_set.add(kw)

        for result in tfidf_results:
            if result['score'] < 0.1:
                continue
            response = result['response']
            for kw in topic_keywords_set:
                if kw in response:
                    response = self._ensure_first_person(response)
                    return self._trim_response(response)

        # 从话题索引中查找
        for topic in topics:
            items = self.kb.topic_index.get(topic, [])
            if not items:
                continue

            user_words = set(jieba.cut(user_input))
            best_item = None
            best_score = 0

            for item in items:
                item_words = set(jieba.cut(item['text']))
                overlap = user_words & item_words
                score = len(overlap)
                if score > best_score:
                    best_score = score
                    best_item = item

            if best_item and best_score >= 2:
                response = self._ensure_first_person(best_item['text'])
                return self._trim_response(response)

        return None

    def _tfidf_search(self, user_input, top_k=5):
        """用 TF-IDF 搜索最相似的内容"""
        if self.vectorizer is None:
            return []

        user_tokenized = ' '.join(jieba.cut(user_input))
        try:
            user_vec = self.vectorizer.transform([user_tokenized])
        except Exception:
            return []

        similarities = cosine_similarity(user_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < 0.05:
                continue
            item = self.search_corpus[idx].copy()
            item['score'] = score
            results.append(item)

        return results

    def _ensure_first_person(self, text):
        """确保回复是第一人称（户晨风视角）"""
        text = re.sub(r'^(某网友|网友|户晨风)[：:]\s*', '', text)
        text = text.replace('户晨风说', '我说')
        text = text.replace('户晨风觉得', '我觉得')
        text = text.replace('户子说', '我说')
        return text

    def _trim_response(self, text, max_length=400):
        """截断过长的回复"""
        if len(text) <= max_length:
            return text

        truncated = text[:max_length]
        last_punct = max(
            truncated.rfind('。'),
            truncated.rfind('！'),
            truncated.rfind('？'),
            truncated.rfind('；'),
        )

        if last_punct > max_length * 0.4:
            return truncated[:last_punct + 1]
        return truncated + '……'

    def _topic_response(self, user_input, topics):
        """根据话题生成户晨风式回答"""
        # 通用话题回复
        general_topic_responses = {
            '经济': [
                '你听我讲啊，经济这个事说白了就是供需关系。你不能老夜郎自大，要实事求是。',
                '经济的事太复杂，我就一普通人。但我知道一点，老百姓的日子得实实在在。',
                '经济好不好不是你说了算，是你口袋里的钱说了算。',
            ],
            '就业': [
                '就业这个事，说难也难，说简单也简单。关键是你想要什么样的生活。',
                '现在这就业环境确实不行。你想找到一个心仪的工作，收入能够维持生存线就不错了。',
                '年轻人最重要的是先养活自己。你别整天想那些有的没的。',
            ],
            '教育': [
                '你手里有个文凭，你真的能省很多事。非常多的事。',
                '学历科的不用担心没饭吃。无论社会发生什么变化，只要英语不错，基本上都有饭吃。',
                '学习这个事，活到老学到老。你得学有用的东西。',
            ],
            '润/移民': [
                '润这个事吧，各人有各人的选择。但是你要实事求是，人家确实要好得多。',
                '你润不润是你的自由。但是你要清楚你在这能得到什么，出去能得到什么。',
            ],
            '科技': [
                '芯片这个事，大概率不可能弯道超车。几百个环节，上千种配套，全是卡脖子的。',
                '任何一个行业的最顶端都是人家的。无一例外。模仿的人永远只能做第二。',
                'AI这个事，好用就行，不好用就不行。你想那么多干嘛？',
            ],
            '医疗': [
                '中医那个副作用未知，这比有明确的副作用更可怕。',
                '看病确实贵。门诊报销比例极低，基本上可以忽略不计。',
            ],
            '社会': [
                '咱就聊聊老百姓关心的事。我就关心衣食住行，吃喝拉撒。',
                '要实事求是，不能含糊其词。事实是什么就是什么。',
                '你不能老夜郎自大。你要实事求是地看问题。',
            ],
            '直播': [
                '直播嘛，就是跟大家聊聊天，挣点辛苦钱。',
                '我这直播就这样，没啥特别的。大家来了就聊聊。',
            ],
            '生活': [
                '生活嘛，柴米油盐，过好每一天就行了。',
                '生活的本质是什么？是衣食住行。你把这几样搞好了，生活就差不了。',
            ],
            '情感': [
                '感情的事，各人有各人的缘法，我不好多说。',
                '你得先把自己过好。你自己都养不活，你拿什么养家？',
            ],
        }

        responses = []
        for topic in topics:
            if topic in general_topic_responses:
                responses.extend(general_topic_responses[topic])

        if responses:
            return random.choice(responses)

        return self._generic_fallback(user_input)

    def _generic_fallback(self, user_input):
        """通用兜底回答 - 保持户晨风风格"""
        fallbacks = [
            '嗯，你说的这个我得想想。我说的不一定对啊，我的愚见。',
            '这个嘛，有点意思。你继续说，我想听听你的观点。',
            '你这个问题本身就不太清晰。你到底想问什么？直接表达观点。',
            '这个事吧，说来话长。改天再聊，今天时间有限。',
            '你说的有道理，但是我有不同看法。你要实事求是，不能只看一面。',
            '不着急，慢慢说。你想聊什么直接说，别铺垫半天。',
            '这个我不太懂，我就不乱说了。咱聊点我了解的。',
            '你听我讲啊，这个事不是你想的那么简单。但是具体怎么个不简单法，我也说不太清楚。',
            '你想表达什么？你直接说你的观点，别绕弯子。',
            '好，下一个。',
        ]
        return random.choice(fallbacks)

    def _record_bot(self, text):
        """记录机器人的回复到历史"""
        self.conversation_history.append({
            'role': 'bot',
            'text': text,
        })

    def get_stats(self):
        """获取机器人统计信息"""
        return {
            'corpus_size': len(self.search_corpus),
            'qa_pairs': len(self.parser.qa_pairs),
            'hu_responses': len(self.parser.hu_responses),
            'total_turns': len(self.parser.conversations),
            'opinions_extracted': len(self.kb.opinion_extractor.opinions),
            'topics_indexed': len(self.kb.topic_index),
            'knowledge_entries': len(self.hu_knowledge.entries),
            'history_length': len(self.conversation_history),
        }

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []

    def search_quotes(self, query, top_k=5):
        """搜索户晨风语录"""
        results = self._tfidf_search(query, top_k=top_k)
        return [
            {
                'text': r['response'][:200],
                'score': r['score'],
                'source': r['source'],
            }
            for r in results
        ]
