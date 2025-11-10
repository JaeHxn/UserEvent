from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel
import sqlite3
from user_events import event_manager
from lightgcn_data_prep import LightGCNDataPreprocessor
import json
import os
from datetime import datetime, timedelta

app = FastAPI(title="관리자 대시보드", version="1.0.0")
templates = Environment(loader=FileSystemLoader("templates"))

# Pydantic 모델 정의
class ResetStatisticsData(BaseModel):
    confirm: bool = False

@app.get('/admin', response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """관리자 대시보드 페이지"""
    template = templates.get_template("admin_dashboard.html")
    return HTMLResponse(template.render(request=request))

@app.get('/api/stats')
async def get_statistics():
    """전체 통계 API"""
    try:
        conn = sqlite3.connect(event_manager.db_path)
        cursor = conn.cursor()
        
        # 전체 사용자 수
        cursor.execute('SELECT COUNT(*) FROM users')
        total_users = cursor.fetchone()[0]
        
        # 전체 상품 조회 수
        cursor.execute('SELECT COUNT(*) FROM product_views')
        total_views = cursor.fetchone()[0]
        
        # 전체 검색 수
        cursor.execute('SELECT COUNT(*) FROM search_events')
        total_searches = cursor.fetchone()[0]
        
        # 오늘 상품 조회 수
        cursor.execute('SELECT COUNT(*) FROM product_views WHERE DATE(timestamp) = DATE("now")')
        today_views = cursor.fetchone()[0]
        
        # 오늘 검색 수
        cursor.execute('SELECT COUNT(*) FROM search_events WHERE DATE(timestamp) = DATE("now")')
        today_searches = cursor.fetchone()[0]
        
        conn.close()
        
        return JSONResponse({
            'status': 'success',
            'data': {
                'total_users': total_users,
                'total_views': total_views,
                'total_searches': total_searches,
                'today_views': today_views,
                'today_searches': today_searches
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/popular_products')
async def get_popular_products(days: int = 7):
    """인기 상품 API"""
    try:
        popular_products = event_manager.get_popular_products(days=days, limit=20)
        return JSONResponse({'status': 'success', 'data': popular_products})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/user_activity')
async def get_user_activity(days: int = 7):
    """사용자 활동 추이 API"""
    try:
        conn = sqlite3.connect(event_manager.db_path)
        cursor = conn.cursor()
        
        # 일별 상품 조회 수
        cursor.execute('''
            SELECT DATE(timestamp) as date, COUNT(*) as view_count
            FROM product_views 
            WHERE timestamp >= DATE("now", '-{} days')
            GROUP BY DATE(timestamp)
            ORDER BY date
        '''.format(days))
        
        daily_views = [{'date': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        # 일별 검색 수 (한국 시간 기준)
        cursor.execute('''
            SELECT strftime('%Y-%m-%d', timestamp) as date, 
                   COUNT(*) as search_count
            FROM search_events 
            WHERE DATE(timestamp) >= DATE("now", '-{} days')
            GROUP BY strftime('%Y-%m-%d', timestamp)
            ORDER BY date
        '''.format(days))
        
        daily_searches = [{'date': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        # 시간별 활동 (오늘)
        cursor.execute('''
            SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
            FROM product_views 
            WHERE DATE(timestamp) = DATE("now")
            GROUP BY strftime('%H', timestamp)
            ORDER BY hour
        ''')
        
        hourly_activity = [{'hour': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        conn.close()
        
        return JSONResponse({
            'status': 'success',
            'data': {
                'daily_views': daily_views,
                'daily_searches': daily_searches,
                'hourly_activity': hourly_activity
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/category_stats')
async def get_category_stats(days: int = None):
    """카테고리별 통계 API"""
    try:
        conn = sqlite3.connect(event_manager.db_path)
        cursor = conn.cursor()
        
        # 기간 필터 조건
        date_filter = ""
        if days:
            date_filter = f"WHERE timestamp >= DATE('now', '-{days} days') AND category != '' AND category IS NOT NULL"
        else:
            date_filter = "WHERE category != '' AND category IS NOT NULL"
        
        # 카테고리별 상품 조회 수
        cursor.execute(f'''
            SELECT category, COUNT(*) as view_count
            FROM product_views 
            {date_filter}
            GROUP BY category
            ORDER BY view_count DESC
            LIMIT 20
        ''')
        
        category_stats = [{'category': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        conn.close()
        
        return JSONResponse({'status': 'success', 'data': category_stats})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/api/reset_statistics')
async def reset_statistics(data: ResetStatisticsData):
    """통계 데이터 초기화 API"""
    try:
        if not data.confirm:
            raise HTTPException(status_code=400, detail='확인 파라미터가 필요합니다.')
        
        conn = sqlite3.connect(event_manager.db_path)
        cursor = conn.cursor()
        
        # 모든 이벤트 데이터 삭제
        cursor.execute('DELETE FROM user_events')
        cursor.execute('DELETE FROM product_views')
        cursor.execute('DELETE FROM search_events')
        cursor.execute('DELETE FROM users')
        
        # AUTOINCREMENT를 사용하지 않으므로 sqlite_sequence 테이블은 존재하지 않음
        # 따라서 시퀀스 리셋은 불필요
        
        conn.commit()
        conn.close()
        
        return JSONResponse({
            'status': 'success', 
            'message': '모든 통계 데이터가 성공적으로 초기화되었습니다.'
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/period_stats')
async def get_period_statistics(days: int = 7):
    """기간별 통계 API"""
    try:
        
        conn = sqlite3.connect(event_manager.db_path)
        cursor = conn.cursor()
        
        # 현재 날짜 가져오기 (한국 시간)
        korea_now = datetime.utcnow() + timedelta(hours=9)
        
        # 기간별 사용자 수
        cursor.execute('''
            SELECT COUNT(DISTINCT user_id) as user_count
            FROM users 
            WHERE created_at >= ?
        ''', (korea_now - timedelta(days=days),))
        
        period_users = cursor.fetchone()[0]
        
        # 기간별 상품 조회 수
        cursor.execute('''
            SELECT COUNT(*) as view_count
            FROM product_views 
            WHERE timestamp >= ?
        ''', (korea_now - timedelta(days=days),))
        
        period_views = cursor.fetchone()[0]
        
        # 기간별 검색 수
        cursor.execute('''
            SELECT COUNT(*) as search_count
            FROM search_events 
            WHERE datetime(timestamp, '+9 hours') >= ?
        ''', (korea_now - timedelta(days=days),))
        
        period_searches = cursor.fetchone()[0]
        
        # 기간별 일일 평균
        daily_avg_views = period_views / days if days > 0 else 0
        daily_avg_searches = period_searches / days if days > 0 else 0
        
        # 기간별 인기 상품
        cursor.execute('''
            SELECT product_code, product_name, COUNT(*) as view_count
            FROM product_views 
            WHERE timestamp >= datetime('now', '-{} days')
            GROUP BY product_code, product_name
            ORDER BY view_count DESC
            LIMIT 10
        '''.format(days))
        
        period_popular_products = [
            {'product_code': row[0], 'product_name': row[1], 'view_count': row[2]} 
            for row in cursor.fetchall()
        ]
        
        # 기간별 인기 카테고리
        cursor.execute('''
            SELECT category, COUNT(*) as view_count
            FROM product_views 
            WHERE timestamp >= datetime('now', '-{} days') AND category != '' AND category IS NOT NULL
            GROUP BY category
            ORDER BY view_count DESC
            LIMIT 10
        '''.format(days))
        
        period_popular_categories = [
            {'category': row[0], 'view_count': row[1]} 
            for row in cursor.fetchall()
        ]
        
        conn.close()
        
        return JSONResponse({
            'status': 'success',
            'data': {
                'period_days': days,
                'period_users': period_users,
                'period_views': period_views,
                'period_searches': period_searches,
                'daily_avg_views': round(daily_avg_views, 2),
                'daily_avg_searches': round(daily_avg_searches, 2),
                'popular_products': period_popular_products,
                'popular_categories': period_popular_categories
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/lightgcn_data')
async def get_lightgcn_data():
    """LightGCN 데이터 상태 확인 API"""
    try:
        preprocessor = LightGCNDataPreprocessor()
        data = preprocessor.load_lightgcn_data()
        
        if not data:
            raise HTTPException(status_code=404, detail='LightGCN 데이터가 없습니다.')
        
        return JSONResponse({
            'status': 'success',
            'data': {
                'n_users': data.get('n_users', 0),
                'n_products': data.get('n_products', 0),
                'n_interactions': data.get('n_interactions', 0),
                'sparsity': data.get('sparsity', 0)
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/regenerate_lightgcn_data')
async def regenerate_lightgcn_data(min_interactions: int = 3):
    """LightGCN 데이터 재생성 API"""
    try:
        preprocessor = LightGCNDataPreprocessor()
        data = preprocessor.prepare_lightgcn_data(min_interactions=min_interactions)
        
        if data:
            return JSONResponse({
                'status': 'success',
                'message': 'LightGCN 데이터가 성공적으로 재생성되었습니다.',
                'data': {
                    'n_users': data['n_users'],
                    'n_products': data['n_products'],
                    'n_interactions': data['n_interactions']
                }
            })
        else:
            raise HTTPException(status_code=400, detail='데이터가 충분하지 않습니다.')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/user_details/{user_id}')
async def get_user_details(user_id: str):
    """사용자 상세 정보 API"""
    try:
        conn = sqlite3.connect(event_manager.db_path)
        cursor = conn.cursor()
        
        # 사용자 기본 정보
        cursor.execute('''
            SELECT user_id, session_id, ip_address, created_at, last_activity
            FROM users WHERE user_id = ?
        ''', (user_id,))
        
        user_info = cursor.fetchone()
        if not user_info:
            raise HTTPException(status_code=404, detail='사용자를 찾을 수 없습니다.')
        
        # 사용자 상품 조회 기록
        cursor.execute('''
            SELECT product_code, product_name, category, price, timestamp
            FROM product_views 
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 20
        ''', (user_id,))
        
        product_views = [
            {
                'product_code': row[0],
                'product_name': row[1],
                'category': row[2],
                'price': row[3],
                'timestamp': row[4]
            }
            for row in cursor.fetchall()
        ]
        
        # 사용자 검색 기록
        cursor.execute('''
            SELECT query, price_min, price_max, results_count, timestamp
            FROM search_events 
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 20
        ''', (user_id,))
        
        search_history = [
            {
                'query': row[0],
                'price_min': row[1],
                'price_max': row[2],
                'results_count': row[3],
                'timestamp': row[4]
            }
            for row in cursor.fetchall()
        ]
        
        conn.close()
        
        return JSONResponse({
            'status': 'success',
            'data': {
                'user_info': {
                    'user_id': user_info[0],
                    'session_id': user_info[1],
                    'ip_address': user_info[2],
                    'created_at': user_info[3],
                    'last_activity': user_info[4]
                },
                'product_views': product_views,
                'search_history': search_history
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    
    # 환경 변수에서 설정 가져오기
    host = '0.0.0.0'
    port = 7071
    debug = True
    
    print(f"🚀 FastAPI 관리자 대시보드 시작: {host}:{port} (debug={debug})")
    uvicorn.run("admin_dashboard:app", host=host, port=port, reload=debug) 