import os
import re
from datetime import datetime
from copy import deepcopy
import json

from tqdm import tqdm


def url2dir_douyin(url, root_path):
    """
    根据抖音URL和根目录获取多媒体保存文件夹

    参数:
    url (str): 抖音视频页面 URL 地址
    root_dir (str): 保存视频的根文件夹路径
    """
    def clean_filename(video_url):
        """
        返回合适的视频文件夹名
        
        参数:
        video_url (str): 视频的原始url
        
        返回:
        str: 清理后的文件夹名
        """
        if video_url[-1] == '/':
            video_url = video_url[:-1]

        return video_url.split('/')[-1]

    return os.path.join(root_path, clean_filename(url), clean_filename(url) + '.mp4')

def url2dir_toutiao(url, root_path):
    """
    根据头条URL和根目录获取多媒体保存文件夹

    参数:
    url (str): 头条页面 URL 地址
    root_dir (str): 保存多媒体的根文件夹路径
    """
    def clean_filename(url):
        """
        返回合适的文件名

        参数:
        url (str): 原始 URL

        返回:
        str: 清理后的文件名
        """
        if url.endswith('/'):
            url = url[:-1]
        return re.sub(r'[^\w\-]', '_', url.split('/')[-1])
    return os.path.join(root_path, clean_filename(url), clean_filename(url) + '.mp4')

def url2dir_xigua(url, root_path):
    def clean_filename(video_url):
        """
        返回合适的视频文件名

        参数:
        video_url (str): 视频的原始 URL

        返回:
        str: 清理后的文件名
        """
        if video_url[-1] == '/':
            video_url = video_url[:-1]
        return re.sub(r'[^\w\-]', '_', video_url.split('/')[-1])
    return os.path.join(root_path, clean_filename(url), clean_filename(url) + '.mp4')

def url2dir_kuaishou(url, root_path):
    """
    根据快手URL和根目录获取多媒体保存文件夹

    参数:
    url (str): 快手页面 URL 地址
    root_dir (str): 保存多媒体的根文件夹路径
    """
    def clean_filename(video_url):
        """
        返回合适的视频文件名

        参数:
        video_url (str): 视频的原始 URL

        返回:
        str: 清理后的文件名
        """
        if video_url[-1] == "/":
            video_url = video_url[:-1]
        return re.sub(r"[^\w\-]", "_", video_url.split("/")[-1])
    return os.path.join(root_path, clean_filename(url), clean_filename(url) + '.mp4')

def url2dir_bilibili(url, root_path):
    def clean_filename(video_url: str) -> str:
        if video_url.endswith('/'):
            video_url = video_url[:-1]
        return video_url.split('/')[-1]
    return os.path.join(root_path, clean_filename(url), clean_filename(url) + '.mp4')

url2dir_platform = {'抖音': url2dir_douyin,
                    '头条': url2dir_toutiao,
                    '西瓜': url2dir_xigua,
                    '快手': url2dir_kuaishou,
                    '哔哩哔哩':url2dir_bilibili}

def find_file(url, platform, root_path='/home/caidesheng/nasdata/media'):
    proot = {'抖音': 'douyin',
             '头条': 'toutiao',
             '西瓜': 'xigua',
             '快手': 'kuaishou',
             '哔哩哔哩': 'bilibili'}
    
    return url2dir_platform[platform](url, os.path.join(root_path, proot[platform]))


def split_dataset(source_path='data/available_dataset.json', target_path="data/splited_dataset.json"):
    '''
    将数据集样本按时间帧分割为多个样本

    参数:
    source_path (str): 源数据集地址
    target_path (str): 目标数据集地址
    '''
    dataset = json.load(open(source_path, encoding='utf8'))
    splited_dataset = {}
    index = 0
    inter_keys = ['粉丝量', '点赞量', '播放量', '分享量', '收藏量', '评论量']
    for video in dataset.values():
        history = [index]
        # 创建刚发布状态
        splited_dataset[index] = deepcopy(video)
        post_date = datetime.strptime(video['发布时间'], "%Y-%m-%d %H:%M:%S")
        post_date  = datetime.strftime(post_date, "%Y-%m-%d")
        splited_dataset[index]['当前时间'] = post_date
        for k in inter_keys:
            splited_dataset[index]['当前' + k] = 0
        splited_dataset[index]['当前粉丝量'] = list(video['互动量信息'].values())[0]['粉丝量']
        del splited_dataset[index]['互动量信息']
        del splited_dataset[index]['评论']
        splited_dataset[index]['评论'] = {}
        splited_dataset[index]['历史状态'] = []
        splited_dataset[index]['最终状态'] = False
        index += 1
        record_dates = list(video['互动量信息'].keys())
        # 创建采集状态
        for date in record_dates:
            splited_dataset[index] = deepcopy(video)
            splited_dataset[index]['当前时间'] = date
            for k in inter_keys:
                splited_dataset[index]['当前' + k] = video['互动量信息'][date][k]
            del splited_dataset[index]['互动量信息']
            del splited_dataset[index]['评论']
            splited_dataset[index]['评论'] = {}
            date = datetime.strptime(date, "%Y-%m-%d").date()
            for cid, c in video['评论'].items():
                # 比较发布时间
                cdate = datetime.strptime(c['评论时间'], "%Y-%m-%d %H:%M:%S").date()
                if cdate <= date:
                    splited_dataset[index]['评论'][cid] = c
            splited_dataset[index]['历史状态'] = deepcopy(history)
            if date == record_dates[-1]:
                splited_dataset[index]['最终状态'] = True
            else:
                splited_dataset[index]['最终状态'] = False
            history.append(index)
            index += 1

    with open(target_path, "w", encoding='utf-8') as f:
        json.dump(splited_dataset, f, ensure_ascii=False, indent=3)


def avai_video_dataset(source_path='data/ranked_dataset.json', target_path="data/available_dataset.json"):
    '''
    筛选出视频存在的样本
    '''
    key_tobe_del = []
    dataset = json.load(open(source_path, encoding='utf8'))
    for vid, data in tqdm(dataset.items()):
        video_path = find_file(data['url'], data['平台'])
        if not os.path.exists(video_path):
            key_tobe_del.append(vid)
        dataset[vid]['视频地址'] = video_path

    for vid in key_tobe_del:
        del dataset[vid]

    print(f"{len(key_tobe_del)} videos are deleted")

    dataset = {i: v for i, v in enumerate(dataset.values())}
    print(f"{len(dataset)} videos left")

    with open(target_path, 'w', encoding='utf8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=3)


if __name__ == '__main__':
    avai_video_dataset(source_path='../ranked_dataset.json', target_path="../available_dataset.json")
    # split_dataset(source_path='data/available_dataset.json', target_path="data/splited_dataset.json")
