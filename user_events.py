import sqlite3
import json
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import hashlib
import os

class UserEventManager:
    def __init__(self, db_path: str = "user_events.db"):
        """사용자 이벤트 관리자 초기화"""
        self.db_path = db_path
        self.init_database()
    
    def get_korean_time(self) -> str:
        """한국 시간을 반환 (UTC+9)"""
        korean_tz = timezone(timedelta(hours=9))
        return datetime.now(korean_tz).strftime('%Y-%m-%d %H:%M:%S')
    
    def init_database(self):
        """데이터베이스 테이블 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 사용자 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                session_id TEXT,
                ip_address TEXT,
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 사용자 이벤트 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_events (
                event_id TEXT PRIMARY KEY,
                user_id TEXT,
                session_id TEXT,
                event_type TEXT,
                product_code TEXT,
                product_name TEXT,
                category TEXT,
                price INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # 상품 상세보기 이벤트 테이블 (LightGCN용)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS product_views (
                view_id TEXT PRIMARY KEY,
                user_id TEXT,
                product_code TEXT,
                product_name TEXT,
                category TEXT,
                price INTEGER,
                view_duration INTEGER DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # 검색 이벤트 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_events (
                search_id TEXT PRIMARY KEY,
                user_id TEXT,
                query TEXT,
                price_min INTEGER,
                price_max INTEGER,
                results_count INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_or_create_user(self, session_id: str, ip_address: str = None, user_agent: str = None) -> str:
        """사용자 ID를 가져오거나 새로 생성"""
        # 세션 ID를 기반으로 사용자 ID 생성 (해시)
        user_id = hashlib.md5(session_id.encode()).hexdigest()[:16]
        korean_time = self.get_korean_time()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 사용자 존재 여부 확인
        cursor.execute('SELECT user_id FROM users WHERE user_id = ?', (user_id,))
        existing_user = cursor.fetchone()
        
        if not existing_user:
            # 새 사용자 생성
            cursor.execute('''
                INSERT INTO users (user_id, session_id, ip_address, user_agent, created_at, last_activity)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, session_id, ip_address, user_agent, korean_time, korean_time))
        else:
            # 마지막 활동 시간 업데이트
            cursor.execute('''
                UPDATE users SET last_activity = ?
                WHERE user_id = ?
            ''', (korean_time, user_id))
        
        conn.commit()
        conn.close()
        return user_id
    
    def record_product_view(self, user_id: str, session_id: str, product_data: Dict[str, Any], 
                           ip_address: str = None, user_agent: str = None) -> str:
        """상품 상세보기 이벤트 기록"""
        event_id = str(uuid.uuid4())
        view_id = str(uuid.uuid4())
        korean_time = self.get_korean_time()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 상품 상세보기 이벤트 기록
        cursor.execute('''
            INSERT INTO product_views (
                view_id, user_id, product_code, product_name, category, 
                price, ip_address, user_agent, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            view_id, user_id, 
            product_data.get('상품코드', ''),
            product_data.get('제목', ''),
            product_data.get('카테고리', ''),
            product_data.get('가격', 0),
            ip_address, user_agent, korean_time
        ))
        
        # 일반 이벤트 테이블에도 기록
        metadata = json.dumps(product_data, ensure_ascii=False)
        cursor.execute('''
            INSERT INTO user_events (
                event_id, user_id, session_id, event_type, product_code,
                product_name, category, price, metadata, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event_id, user_id, session_id, 'product_view',
            product_data.get('상품코드', ''),
            product_data.get('제목', ''),
            product_data.get('카테고리', ''),
            product_data.get('가격', 0),
            metadata, korean_time
        ))
        
        conn.commit()
        conn.close()
        return view_id
    
    def record_search_event(self, user_id: str, session_id: str, query: str, 
                           price_min: int = None, price_max: int = None,
                           results_count: int = 0, ip_address: str = None, 
                           user_agent: str = None) -> str:
        """검색 이벤트 기록"""
        search_id = str(uuid.uuid4())
        event_id = str(uuid.uuid4())
        korean_time = self.get_korean_time()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 검색 이벤트 기록
        cursor.execute('''
            INSERT INTO search_events (
                search_id, user_id, query, price_min, price_max,
                results_count, ip_address, user_agent, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            search_id, user_id, query, price_min, price_max,
            results_count, ip_address, user_agent, korean_time
        ))
        
        # 일반 이벤트 테이블에도 기록
        metadata = json.dumps({
            'query': query,
            'price_min': price_min,
            'price_max': price_max,
            'results_count': results_count
        }, ensure_ascii=False)
        
        cursor.execute('''
            INSERT INTO user_events (
                event_id, user_id, session_id, event_type, metadata, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (event_id, user_id, session_id, 'search', metadata, korean_time))
        
        conn.commit()
        conn.close()
        return search_id
    
    def get_user_product_views(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """사용자의 상품 조회 기록 가져오기"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT product_code, product_name, category, price, timestamp
            FROM product_views 
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (user_id, limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'product_code': row[0],
                'product_name': row[1],
                'category': row[2],
                'price': row[3],
                'timestamp': row[4]
            })
        
        conn.close()
        return results
    
    def get_user_search_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """사용자의 검색 기록 가져오기"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT query, price_min, price_max, results_count, timestamp
            FROM search_events 
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (user_id, limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'query': row[0],
                'price_min': row[1],
                'price_max': row[2],
                'results_count': row[3],
                'timestamp': row[4]
            })
        
        conn.close()
        return results
    
    def get_popular_products(self, days: int = 7, limit: int = 20) -> List[Dict[str, Any]]:
        """인기 상품 조회 (LightGCN용)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # days가 None이거나 0 이하인 경우 전체 기간 조회
        if days is None or days <= 0:
            cursor.execute('''
                SELECT product_code, product_name, category, price, COUNT(*) as view_count
                FROM product_views 
                GROUP BY product_code
                ORDER BY view_count DESC
                LIMIT ?
            ''', (limit,))
        else:
            cursor.execute('''
                SELECT product_code, product_name, category, price, COUNT(*) as view_count
                FROM product_views 
                WHERE timestamp >= datetime('now', '-{} days')
                GROUP BY product_code
                ORDER BY view_count DESC
                LIMIT ?
            '''.format(days), (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'product_code': row[0],
                'product_name': row[1],
                'category': row[2],
                'price': row[3],
                'view_count': row[4]
            })
        
        conn.close()
        return results
    
    def get_user_interaction_matrix(self) -> Dict[str, Any]:
        """LightGCN용 사용자-상품 상호작용 행렬 생성"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 사용자별 상품 조회 데이터
        cursor.execute('''
            SELECT user_id, product_code, COUNT(*) as interaction_count
            FROM product_views
            GROUP BY user_id, product_code
        ''')
        
        user_product_interactions = {}
        for row in cursor.fetchall():
            user_id, product_code, count = row
            if user_id not in user_product_interactions:
                user_product_interactions[user_id] = {}
            user_product_interactions[user_id][product_code] = count
        
        # 고유 사용자와 상품 목록
        cursor.execute('SELECT DISTINCT user_id FROM users')
        unique_users = [row[0] for row in cursor.fetchall()]
        
        cursor.execute('SELECT DISTINCT product_code FROM product_views')
        unique_products = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            'user_product_interactions': user_product_interactions,
            'unique_users': unique_users,
            'unique_products': unique_products
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """데이터베이스 통계 정보"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 총 사용자 수
        cursor.execute('SELECT COUNT(*) FROM users')
        total_users = cursor.fetchone()[0]
        
        # 총 상품 조회 수
        cursor.execute('SELECT COUNT(*) FROM product_views')
        total_views = cursor.fetchone()[0]
        
        # 총 검색 수
        cursor.execute('SELECT COUNT(*) FROM search_events')
        total_searches = cursor.fetchone()[0]
        
        # 오늘 상품 조회 수
        cursor.execute('''
            SELECT COUNT(*) FROM product_views 
            WHERE DATE(timestamp) = DATE('now')
        ''')
        today_views = cursor.fetchone()[0]
        
        # 오늘 검색 수
        cursor.execute('''
            SELECT COUNT(*) FROM search_events 
            WHERE DATE(timestamp) = DATE('now')
        ''')
        today_searches = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_users': total_users,
            'total_views': total_views,
            'total_searches': total_searches,
            'today_views': today_views,
            'today_searches': today_searches
        }

# 전역 인스턴스
event_manager = UserEventManager() 