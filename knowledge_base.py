# -*- coding: utf-8 -*-
"""
户晨风对话机器人 - 观点知识库
从对话中提取户晨风在各话题上的立场和观点
"""
import re
import logging
from typing import List, Dict, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class OpinionExtractor:
    """从对话记录中提取户晨风的观点"""
    
    def __init__(self):
        self.opinions = []  # 所有提取的观点
        self.topic_opinions = defaultdict(list)  # 按话题分类的观点
    
    def extract_from_conversations(self, conversations: List[Dict]) -> List[Dict]:
        """
        从对话流中提取户晨风的观点段落
        核心逻辑：找到户晨风在回答某个话题时说的有观点性的长段落
        """
        opinions = []
        
        for i, turn in enumerate(conversations):
            if turn['speaker'] != '户晨风':
                continue
            
            text = turn['text']
            
            # 过滤太短的回复（通常不是观点）
            if len(text) < 50:
                continue
            
            # 提取观点段落
            opinion = self._analyze_opinion(text, turn)
            if opinion:
                opinions.append(opinion)
        
        self.opinions = opinions
        logger.info("提取观点段落: {} 条".format(len(opinions)))
        
        # 对话题分类
        self._classify_by_topic(opinions)
        
        return opinions
    
    def _analyze_opinion(self, text: str, turn: Dict) -> Dict:
        """分析一段话是否包含观点，并提取关键信息"""
        opinion = {
            'text': text,
            'date': turn.get('date', ''),
            'month': turn.get('month', ''),
            'topics': [],
            'keywords': [],
            'stance': '',  # 立场摘要
        }
        
        # 检测话题关键词
        from data_parser import get_topic_keywords
        topic_kws = get_topic_keywords()
        
        for topic, keywords in topic_kws.items():
            for kw in keywords:
                if kw in text:
                    opinion['topics'].append(topic)
                    opinion['keywords'].extend([k for k in keywords if k in text])
                    break
        
        # 如果没有匹配到任何话题，尝试提取核心观点
        if not opinion['topics']:
            # 提取"我觉得"、"我认为"、"说白了"等观点标记后的内容
            stance_patterns = [
                r'(?:我觉得|我认为|在我看来|说白了|就是说|归根结底|本质上)[，,]?\s*(.{10,100})',
                r'(?:我的观点|我的态度|我的意见|我的建议)[是]?[：:]?\s*(.{10,100})',
            ]
            for pattern in stance_patterns:
                match = re.search(pattern, text)
                if match:
                    opinion['stance'] = match.group(1).strip()
                    break
        
        # 提取关键观点句
        opinion['key_sentences'] = self._extract_key_sentences(text)
        
        return opinion
    
    def _extract_key_sentences(self, text: str) -> List[str]:
        """从段落中提取关键观点句"""
        sentences = re.split(r'[。！？\n]', text)
        key_sentences = []
        
        opinion_markers = [
            '我觉得', '我认为', '说白了', '就是说', '归根结底',
            '本质上', '我的观点', '我的态度', '我认为', '在我看来',
            '实际上', '事实', '道理', '关键', '核心', '本质',
            '应该', '不应该', '能', '不能', '可以', '不可以',
            '好', '坏', '对', '错', '行', '不行',
        ]
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue
            for marker in opinion_markers:
                if marker in sent:
                    key_sentences.append(sent)
                    break
        
        return key_sentences[:5]  # 最多5个关键句
    
    def _classify_by_topic(self, opinions: List[Dict]):
        """将观点按话题分类"""
        for opinion in opinions:
            for topic in opinion['topics']:
                self.topic_opinions[topic].append(opinion)
        
        for topic, ops in self.topic_opinions.items():
            logger.info("  话题 [{}]: {} 条观点".format(topic, len(ops)))
    
    def get_opinions_for_topic(self, topic: str) -> List[Dict]:
        """获取某个话题下的所有观点"""
        return self.topic_opinions.get(topic, [])
    
    def get_all_topics(self) -> List[str]:
        """获取所有话题"""
        return list(self.topic_opinions.keys())


class KnowledgeBase:
    """户晨风知识库 - 组织和管理提取的知识"""
    
    def __init__(self):
        self.opinion_extractor = OpinionExtractor()
        self.qa_index = []       # 问答对索引
        self.topic_index = {}    # 话题 -> 相关回复
        self.personal_info = {}  # 户晨风的个人信息（用于角色扮演）
        
        # 户晨风的经典语录和口头禅（用于风格化）
        self.catchphrases = [
            "好，下一个。",
            "生活愉快。",
            "请直接表达观点。",
            "你这个观点本身就有问题。",
            "你听我讲啊。",
            "说白了就是这么回事。",
            "我说的都是字面意思。",
            "你要实事求是。",
            "你懂我的意思吧？",
            "咱们不说别的。",
            "我跟你讲。",
            "这个事吧。",
            "你想啊。",
            "对不对？",
            "是不是这个理？",
            "我不一定对啊，我的愚见。",
            "我先免责声明，我说的都不对。",
            "我就是个直播的，聊聊天。",
            "大家晚上也无聊，听我唠一唠。",
            "感谢大家的礼物。",
        ]
        
        # 户晨风的个人信息（从对话中提取的）
        self.personal_info = {
            'name': '户晨风',
            'aliases': ['户子', '胡子'],
            'profession': '主播/内容创作者',
            'platform': 'B站、斗崴（抖音）',
            'location': '成都（经常去各地拍视频）',
            'background': '做过私募基金研究',
            'style': '直来直去，实事求是，接地气',
            'income': '直播一个月两三万',
            'education_view': '鼓励考研考博，认为学历有用但要看专业',
            'emigration_view': '支持有条件的人润出去，认为护理、焊工等好润',
            'tech_view': '认为芯片等核心技术差距巨大，不可能弯道超车',
            'medical_view': '认为中医不靠谱，支持医疗改革',
            'social_view': '关注底层民生，批评社会不公',
        }
    
    def build(self, conversations: List[Dict], qa_pairs: List[Dict]):
        """构建知识库"""
        logger.info("正在构建知识库...")
        
        # 1. 提取观点
        self.opinion_extractor.extract_from_conversations(conversations)
        
        # 2. 构建问答索引
        self._build_qa_index(qa_pairs)
        
        # 3. 构建话题索引
        self._build_topic_index(conversations)
        
        logger.info("知识库构建完成")
    
    def _build_qa_index(self, qa_pairs: List[Dict]):
        """构建问答索引"""
        self.qa_index = qa_pairs
        logger.info("问答索引: {} 条".format(len(qa_pairs)))
    
    def _build_topic_index(self, conversations: List[Dict]):
        """构建话题索引 - 将户晨风的回复按话题归类"""
        from data_parser import get_topic_keywords
        topic_kws = get_topic_keywords()
        
        self.topic_index = defaultdict(list)
        
        for i, turn in enumerate(conversations):
            if turn['speaker'] != '户晨风':
                continue
            text = turn['text']
            if len(text) < 30:
                continue
            
            for topic, keywords in topic_kws.items():
                for kw in keywords:
                    if kw in text:
                        # 收集上下文（网友说了什么）
                        context = ''
                        for j in range(max(0, i-2), i):
                            if conversations[j]['speaker'] != '户晨风':
                                context = conversations[j]['text']
                                break
                        
                        self.topic_index[topic].append({
                            'text': text,
                            'context': context,
                            'date': turn.get('date', ''),
                        })
                        break
        
        for topic, items in self.topic_index.items():
            logger.info("  话题索引 [{}]: {} 条".format(topic, len(items)))
    
    def query(self, question: str, topics: List[str] = None) -> Dict:
        """
        查询知识库，返回户晨风会怎么回答
        
        Returns:
            {
                'answer': str,  # 回答内容
                'source': str,  # 来源
                'confidence': float,  # 置信度
                'topic': str,  # 匹配的话题
            }
        """
        # 1. 先尝试问答对精确匹配
        qa_result = self._search_qa(question)
        if qa_result and qa_result['confidence'] > 0.6:
            return qa_result
        
        # 2. 尝试话题匹配
        if topics:
            topic_result = self._search_by_topic(question, topics)
            if topic_result:
                return topic_result
        
        # 3. 尝试观点匹配
        opinion_result = self._search_opinions(question)
        if opinion_result:
            return opinion_result
        
        # 4. 兜底
        return {
            'answer': '',
            'source': 'none',
            'confidence': 0,
            'topic': '',
        }
    
    def _search_qa(self, question: str) -> Dict:
        """在问答对中搜索"""
        import jieba
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        if not self.qa_index:
            return None
        
        # 简化版：用关键词匹配
        question_words = set(jieba.cut(question))
        best_match = None
        best_score = 0
        
        # 只采样一部分加速
        sample_size = min(5000, len(self.qa_index))
        import random
        sample = random.sample(self.qa_index, sample_size)
        
        for qa in sample:
            q = qa['question']
            # 计算问题相似度
            q_words = set(jieba.cut(q))
            overlap = question_words & q_words
            if not overlap:
                continue
            
            score = len(overlap) / max(len(question_words), 1)
            
            # 额外加分：如果问题中包含相同的关键词
            for w in overlap:
                if len(w) >= 2:
                    score += 0.1
            
            if score > best_score:
                best_score = score
                best_match = qa
        
        if best_match and best_score > 0.2:
            return {
                'answer': best_match['answer'],
                'source': 'qa_pair',
                'confidence': min(best_score, 1.0),
                'topic': '',
                'question_matched': best_match['question'],
            }
        
        return None
    
    def _search_by_topic(self, question: str, topics: List[str]) -> Dict:
        """按话题搜索"""
        import random
        
        for topic in topics:
            items = self.topic_index.get(topic, [])
            if not items:
                continue
            
            # 从该话题的回复中随机选一条有代表性的
            # 优先选较长的（通常更有内容）
            sorted_items = sorted(items, key=lambda x: len(x['text']), reverse=True)
            top_items = sorted_items[:min(20, len(sorted_items))]
            selected = random.choice(top_items)
            
            return {
                'answer': selected['text'],
                'source': 'topic_index',
                'confidence': 0.5,
                'topic': topic,
            }
        
        return None
    
    def _search_opinions(self, question: str) -> Dict:
        """在观点段落中搜索"""
        import jieba
        import random
        
        question_words = set(jieba.cut(question))
        best_opinion = None
        best_score = 0
        
        for opinion in self.opinion_extractor.opinions:
            text_words = set(jieba.cut(opinion['text']))
            overlap = question_words & text_words
            
            if not overlap:
                continue
            
            score = len(overlap) / max(len(question_words), 1)
            
            # 话题匹配加分
            for topic in opinion['topics']:
                if topic in question:
                    score += 0.3
            
            if score > best_score:
                best_score = score
                best_opinion = opinion
        
        if best_opinion and best_score > 0.15:
            # 从观点中提取关键句
            text = best_opinion['text']
            if best_opinion.get('key_sentences'):
                # 用关键句组成回答
                answer = '。'.join(best_opinion['key_sentences'][:3]) + '。'
            else:
                answer = text
            
            return {
                'answer': answer,
                'source': 'opinion',
                'confidence': min(best_score, 0.8),
                'topic': ', '.join(best_opinion['topics']),
            }
        
        return None
