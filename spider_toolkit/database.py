"""
数据库模块 - 处理爬取历史和数据存储
"""

import sqlite3
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from .config import config


class CrawlHistory:
    """爬取历史管理类"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.get('database.db_path', 'crawler_history.db')
        self.enabled = config.database_enabled
        self.logger = logging.getLogger(__name__)
        
        if self.enabled:
            self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS crawl_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        url TEXT NOT NULL,
                        url_hash TEXT UNIQUE NOT NULL,
                        crawl_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        item_count INTEGER DEFAULT 0,
                        file_size INTEGER DEFAULT 0,
                        error_message TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS crawl_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        history_id INTEGER,
                        data_type TEXT NOT NULL,
                        data_content TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (history_id) REFERENCES crawl_history (id)
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_url_hash ON crawl_history(url_hash)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_crawl_type ON crawl_history(crawl_type)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_created_at ON crawl_history(created_at)
                ''')
                
                conn.commit()
                
        except sqlite3.Error as e:
            self.logger.error(f"初始化数据库失败: {e}")
    
    def _get_url_hash(self, url: str) -> str:
        """生成URL的哈希值"""
        return hashlib.md5(url.encode('utf-8')).hexdigest()
    
    def add_record(self, url: str, crawl_type: str, status: str = 'started',
                   item_count: int = 0, file_size: int = 0, 
                   error_message: Optional[str] = None) -> Optional[int]:
        """添加爬取记录"""
        if not self.enabled:
            return None
        
        try:
            url_hash = self._get_url_hash(url)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    INSERT OR REPLACE INTO crawl_history 
                    (url, url_hash, crawl_type, status, item_count, file_size, error_message, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (url, url_hash, crawl_type, status, item_count, file_size, error_message))
                
                conn.commit()
                return cursor.lastrowid
                
        except sqlite3.Error as e:
            self.logger.error(f"添加爬取记录失败: {e}")
            return None
    
    def update_record(self, record_id: int, status: str, 
                     item_count: Optional[int] = None,
                     file_size: Optional[int] = None,
                     error_message: Optional[str] = None):
        """更新爬取记录"""
        if not self.enabled or not record_id:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 构建更新语句
                updates = ['status = ?', 'updated_at = CURRENT_TIMESTAMP']
                params = [status]
                
                if item_count is not None:
                    updates.append('item_count = ?')
                    parmms.append(item_count)
                
                if file_size is not None:
                    updates.append('file_size = ?')
                    params.append(file_size)
                
                if error_message is not None:
                    updates.append('error_message = ?')
                    params.append(error_message)
                
                params.append(record_id)
                
                conn.execute(f'''
                    UPDATE crawl_history 
                    SET {', '.join(updates)}
                    WHERE id = ?
                ''', paramELECT * FROM crawl_history WHERE 1=1'
                params = []
                
                if url:
                    query += ' AND url = ?'
                    params.append(url)
                
                if crawl_type:
                    query += ' AND crawl_type = ?'
                    params.append(crawl_type)
                
                query += ' ORDER BY created_at DESC LIMIT ?'
                params.append(limit)
                
                cursor = conn.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
                
        except sqlite3.Error as e:
                    SELECT * FROM crawl_history 
                    WHERE url_hash = ? AND crawl_type = ?
                    ORDER BY created_at DESC LIMIT 1
                ''', (url_hash, crawl_type))
                
                row = cursor.fetchone()
                return dict(row) if row else None
                
        except sqlite3.Error as e:
            self.logger.error(f"获取爬取记录失败: {e}")
            return None
    
    def is_recently_crawled(self, url: str, crawl_type: str, 
                           hours: int = 24) -> bool:
        """检查是否最近已爬取过"""
        if not self.enabled:
            return False
        
        try:
            url_hash = self._get_url_hash(url)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT COUNT(*) FROM crawl_history 
                    WHERE url_hash = ? AND crawl_type = ? AND status = 'completed'
                    AND datetime(created_at) > datetime('now', '-{} hours')
                '''.format(hours), (url_hash, crawl_type))
                
                count = cursor.fetchone()[0]
                return count > 0
                
        except sqlite3.Error as e:
            self.logger.error(f"检查爬data, ensure_ascii=False)
            
            with sqlite3.connect(self.db_path) as
    def save_data(self, history_id: int, data_type: str, data: Any):
        """保存爬取数据"""
        if not self.enabled or not history_id:
            return
        
        try:
            data_content = json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO crawl_data (history_id, data_type, data_content)
                    VALUES (?, ?, ?)
                ''', (history_id, data_type, data_content))
                
                conn.commit()
                
        except sqlite3.Error as e:
            self.logger.error(f"保存爬取数据失败: {e}")
    
    def get_data(self, history_id: int, data_type: Optional[str] = None) -> List[Dict]:
        """获取爬取数据"""
        if not self.enabled:
            return []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                if data_type:
                    cursor = conn.execute('''
                        SELECT * FROM crawl_data 
                        WHERE history_id = ? AND data_type = ?
                        ORDER BY created_at
                    ''', (history_id, data_type))ta_json'])
                        results.append({
                            'id': row['id'],
                            'data_type': row['data_type'],
                            'data': data,
                            'created_at': row['created_at']
                        })
                    except json.JSONDecodeError:
                        continue
                
                return results
                
        except sqlite3.Error as e:
            self.logger.error(f"获取爬取数据失败: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if n        results.append(row_dict)
                
                return results
                
        except sqlite3.Error as e:
            self.logger.error(f"获取爬取数据失败: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取爬取统计信息"""
        if not self.enabled:
            return {}
                    FROM crawl_history 
                    GROUP BY crawl_type
                ''')
                by_type = dict(cursor.fetchall())
                
                # 按状态统计
                cursor = conn.execute('''
                    SELECT status, COUNT(*) as count 
                    FROM crawl_history 
                    GROUP BY status
                ''')
                by_status = dict(cursor.fetchall())
                
                # 最近7天的记录数
                cursor = conn.execute('''
                    SELECT COUNT(*) FROM crawl_history 
                    WHERE datetime(created_at) > datetime('now', '-7 days')
                ''')
                recent_records = cursor.fetchone()[0]
                
                # 总数据项数
                        SUM(item_count) as items,
                        SUM(file_size) as file_size
                    FROM crawl_history
                    GROUP BY crawl_type
                ''')
                
                stats['by_type'] = {}
                for row in cursor.fetchall():
                    stats['by_type'][row[0]] = {
                        'count': row[1],
                        'items': row[2] or 0,
                        'file_size': row[3] or 0
                    }
                
                # 最近活动
                    WHERE history_id NOT IN (SELECT id FROM crawl_history)
                ''')
                
                # 清理数据库
                cursor.execute('VACUUM')
                
                self.logger.info(f"清理了 {deleted_count} 条旧记录")
                
        except sqlite3.Error as e:
            self.logger.error(f"清理旧记录失败: {e}")
    
    def export_data(self, output_file: str, crawl_type: Optional[str] = None,
                   start_date: Optional[str] = None, end_date: Optional[str] = None):
        """导出数据"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                c""
        if not self.enabled:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 删除旧的数据记录
                conn.execute('''
                    DELETE FROM crawl_data 
                    WHERE history_id IN (
                        SELECT id FROM crawl_history 
                        WHERE datetime(created_at) < datetime('now', '-{} days')
                    )
                '''.format(days))
                
                # 删除旧的历史记录
                cursor = conn.execute('''
                    DELETE FROM crawl_history 
                    WHERE datetime(created_at) < datetime('now', '-{} days')
                '''.format(days))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                self.logger.info(f"清理了 {deleted_count} 条旧记录")
                
        except sqlite3.Error as e:
            self.logger.error(f"清理旧记录失败: {e}")
    
    def export_data(self, output_file: str, crawl_type: Optional[str] = None,
                   start_date: Optional[str] = None, end_date: Optional[str] = None):
        """导出数据"""
        if not self.enabled:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # 构建查询条件
                conditions = []
                params = []
                
                if crawl_type:
                    conditions.append('crawl_type = ?')
                    params.append(crawl_type)
                
                if start_date:
                    conditions.append('date(created_at) >= ?')
                    params.append(start_date)
                
                if end_date:
                    conditions.append('date(created_at) <= ?')
                    params.append(end_date)
                
                where_clause = ' AND '.join(conditions) if conditions else '1=1'
                
                cursor = conn.execute(f'''
                    SELECT h.*, d.data_type, d.data_content
                    FROM crawl_history h
                    LEFT JOIN crawl_data d ON h.id = d.history_id
                    WHERE {where_clause}
                    ORDER BY h.created_at DESC
                ''', params)
                
                rows = cursor.fetchall()
                
                # 转换为字典列表
                data = []
                for row in rows:
                    row_dict = dict(row)
                    if row_dict['data_content']:
                        try:
                            row_dict['data_content'] = json.loads(row_dict['data_content'])
                        except json.JSONDecodeError:
                            pass
                    data.append(row_dict)
                
                # 保存到文件
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str)
                
                self.logger.info(f"数据已导出到: {output_file}")
                
        except sqlite3.Error as e:
            self.logger.error(f"导出数据失败: {e}")


# 全局历史管理实例
history = CrawlHistory()