# -*- coding: utf-8 -*-
"""
户晨风对话机器人 - 数据解析模块
负责从 Markdown 文件中提取对话内容
"""
import os
import re
import json
import logging
from typing import List, Dict, Tuple, Optional

from config import DATA_DIR, HOST_SPEAKER, CALLER_SPEAKERS, MIN_RESPONSE_LENGTH, CACHE_DIR

logger = logging.getLogger(__name__)


class ConversationParser:
    """对话解析器 - 从 md 文件中提取户晨风的对话"""
    
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or DATA_DIR
        self.conversations = []  # 所有对话轮次
        self.hu_responses = []   # 户晨风的回复语料库
        self.qa_pairs = []       # 问答对
        
    def parse_all_files(self) -> List[Dict]:
        """解析所有月份目录下的 md 文件"""
        all_conversations = []
        
        if not os.path.exists(self.data_dir):
            logger.error("数据目录不存在: {}".format(self.data_dir))
            return all_conversations
        
        # 遍历所有年月目录
        for dir_name in sorted(os.listdir(self.data_dir)):
            dir_path = os.path.join(self.data_dir, dir_name)
            if not os.path.isdir(dir_path):
                continue
            if not re.match(r'\d{4}年\d{2}月', dir_name):
                continue
                
            logger.info("正在解析目录: {}".format(dir_name))
            
            # 遍历目录下的 md 文件
            for file_name in sorted(os.listdir(dir_path)):
                if not file_name.endswith('.md') or file_name == 'README.md':
                    continue
                    
                file_path = os.path.join(dir_path, file_name)
                try:
                    convs = self._parse_single_file(file_path, dir_name, file_name)
                    all_conversations.extend(convs)
                    logger.info("  解析文件: {} -> {} 轮对话".format(file_name, len(convs)))
                except Exception as e:
                    logger.warning("  解析文件失败 {}: {}".format(file_name, e))
        
        self.conversations = all_conversations
        logger.info("总共解析 {} 轮对话".format(len(all_conversations)))
        
        # 提取语料库
        self._extract_corpus()
        
        return all_conversations
    
    def _parse_single_file(self, file_path: str, dir_name: str, file_name: str) -> List[Dict]:
        """解析单个 md 文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.strip().split('\n')
        turns = []
        current_speaker = None
        current_text = []
        date = file_name.replace('.md', '')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检测说话者
            speaker_match = re.match(r'^(户晨风|某网友|网友)[：:](.*)', line)
            
            if speaker_match:
                # 保存上一轮
                if current_speaker and current_text:
                    text = ' '.join(current_text).strip()
                    turns.append({
                        'speaker': current_speaker,
                        'text': text,
                        'date': date,
                        'month': dir_name
                    })
                
                current_speaker = speaker_match.group(1)
                current_text = [speaker_match.group(2)]
            else:
                # 继续当前说话者的内容
                if current_speaker:
                    current_text.append(line)
        
        # 保存最后一轮
        if current_speaker and current_text:
            text = ' '.join(current_text).strip()
            turns.append({
                'speaker': current_speaker,
                'text': text,
                'date': date,
                'month': dir_name
            })
        
        return turns
    
    def _extract_corpus(self):
        """从对话中提取语料库和问答对"""
        hu_responses = []
        qa_pairs = []
        
        for i, turn in enumerate(self.conversations):
            # 收集户晨风的所有回复
            if turn['speaker'] == HOST_SPEAKER:
                text = turn['text']
                if len(text) >= MIN_RESPONSE_LENGTH:
                    hu_responses.append({
                        'text': text,
                        'date': turn['date'],
                        'month': turn['month'],
                        'index': i
                    })
            
            # 提取问答对: 网友提问 -> 户晨风回答
            if turn['speaker'] in CALLER_SPEAKERS:
                # 查找后续的户晨风回复
                for j in range(i + 1, min(i + 4, len(self.conversations))):
                    next_turn = self.conversations[j]
                    if next_turn['speaker'] == HOST_SPEAKER:
                        question = turn['text']
                        answer = next_turn['text']
                        if len(answer) >= MIN_RESPONSE_LENGTH:
                            qa_pairs.append({
                                'question': question,
                                'answer': answer,
                                'date': next_turn['date'],
                                'month': next_turn['month']
                            })
                        break
        
        self.hu_responses = hu_responses
        self.qa_pairs = qa_pairs
        logger.info("提取户晨风回复: {} 条".format(len(hu_responses)))
        logger.info("提取问答对: {} 条".format(len(qa_pairs)))
    
    def extract_contextual_pairs(self, window_size=3) -> List[Dict]:
        """提取带上下文的对话对"""
        contextual_pairs = []
        
        for i, turn in enumerate(self.conversations):
            if turn['speaker'] != HOST_SPEAKER:
                continue
            if len(turn['text']) < MIN_RESPONSE_LENGTH:
                continue
            
            # 收集上下文
            context = []
            for j in range(max(0, i - window_size), i):
                prev_turn = self.conversations[j]
                context.append({
                    'speaker': prev_turn['speaker'],
                    'text': prev_turn['text']
                })
            
            # 收集后续回应
            follow_ups = []
            for j in range(i + 1, min(i + window_size + 1, len(self.conversations))):
                next_turn = self.conversations[j]
                follow_ups.append({
                    'speaker': next_turn['speaker'],
                    'text': next_turn['text']
                })
            
            contextual_pairs.append({
                'context': context,
                'response': turn['text'],
                'follow_ups': follow_ups,
                'date': turn['date'],
                'month': turn['month']
            })
        
        logger.info("提取带上下文的对话对: {} 条".format(len(contextual_pairs)))
        return contextual_pairs
    
    def save_corpus(self, output_dir=None):
        """保存提取的语料库到文件"""
        output_dir = output_dir or CACHE_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存户晨风回复语料
        with open(os.path.join(output_dir, 'hu_responses.json'), 'w', encoding='utf-8') as f:
            json.dump(self.hu_responses, f, ensure_ascii=False, indent=2)
        
        # 保存问答对
        with open(os.path.join(output_dir, 'qa_pairs.json'), 'w', encoding='utf-8') as f:
            json.dump(self.qa_pairs, f, ensure_ascii=False, indent=2)
        
        # 保存完整对话
        with open(os.path.join(output_dir, 'conversations.json'), 'w', encoding='utf-8') as f:
            json.dump(self.conversations, f, ensure_ascii=False, indent=2)
        
        logger.info("语料库已保存到: {}".format(output_dir))
    
    def load_corpus(self, cache_dir=None):
        """从缓存加载语料库"""
        cache_dir = cache_dir or CACHE_DIR
        
        responses_path = os.path.join(cache_dir, 'hu_responses.json')
        qa_path = os.path.join(cache_dir, 'qa_pairs.json')
        conv_path = os.path.join(cache_dir, 'conversations.json')
        
        if os.path.exists(responses_path):
            with open(responses_path, 'r', encoding='utf-8') as f:
                self.hu_responses = json.load(f)
        
        if os.path.exists(qa_path):
            with open(qa_path, 'r', encoding='utf-8') as f:
                self.qa_pairs = json.load(f)
        
        if os.path.exists(conv_path):
            with open(conv_path, 'r', encoding='utf-8') as f:
                self.conversations = json.load(f)
        
        logger.info("从缓存加载语料: 回复 {} 条, 问答对 {} 条".format(
            len(self.hu_responses), len(self.qa_pairs)))


def get_topic_keywords() -> Dict[str, List[str]]:
    """定义话题关键词，用于话题分类"""
    return {
        '经济': ['经济', 'GDP', 'gdp', '增长', '衰退', '通胀', '物价', '房价', '房地产', '金融', '银行', '货币', '存钱', '理财', '消费', '省钱', '攒钱', '买房', '租房', '房贷', '按揭', '创业', '做生意', '开店'],
        '就业': ['工作', '就业', '失业', '工资', '收入', '薪资', '加班', '996', '打工', '上班', '职场', '面试', '私企', '外企', '国企', '考编', '公务员', '编制', '辞职', '离职', '跳槽', '换工作', '工厂', '制造业', '进厂', '流水线', '求职', '找工作', '内卷', '躺平', '年轻人', '大学生', '毕业生', '应届', '努力', '奋斗'],
        '教育': ['高考', '考研', '大学', '学历', '本科', '硕士', '博士', '教育', '学校', '专业', '学习', '考试', '考公', '考编', '文凭', '文科', '理科', '英语', '雅思', '托福', '自学', '提升', 'PhD', '科研', '高中', '选专业'],
        '润/移民': ['润', '移民', '出国', '外国', '澳洲', '美国', '日本', '加拿大', '签证', '护照', '绿卡', '雅思', '新西兰', '澳大利亚', '护士', '护理', '焊工', '蓝领', '移居'],
        '直播': ['直播', '开播', '下播', 'PK', '连麦', '弹幕', '礼物', '粉丝', '主播', '平台', 'B站', '斗崴', '切片', '转发', '视频', '拍摄', '内容', '创作', '助理', '招聘'],
        '社会': ['社会', '民生', '政府', '政策', '制度', '公平', '正义', '底层', '老百姓', '人民', '实事求是', '夜郎自大', '战争', '冲突', '侵略', '国际'],
        '生活': ['吃饭', '睡觉', '生活', '日常', '购物', '超市', '买菜', '做饭', '衣服', '酒店', '全季', '焦虑', '迷茫', '选择', '成功', '人生赢家', '过日子'],
        '科技': ['芯片', '科技', '技术', '华为', '手机', '电脑', '互联网', 'AI', '人工智能', '半导体', '光刻机', '国产', 'ChatGPT', '大模型', '程序员', '创新'],
        '医疗': ['医院', '医生', '护士', '看病', '医疗', '医保', '医药', '护理', '中医', '中药', '门诊', '报销', '针灸', '推拿', '社保', '医疗保险', '夜班'],
        '情感': ['结婚', '离婚', '恋爱', '感情', '对象', '相亲', '彩礼', '婚姻', '分手', '相亲'],
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = ConversationParser()
    parser.parse_all_files()
    parser.save_corpus()
    print("解析完成!")
    print("户晨风回复: {} 条".format(len(parser.hu_responses)))
    print("问答对: {} 条".format(len(parser.qa_pairs)))
    print("总对话轮次: {} 条".format(len(parser.conversations)))
