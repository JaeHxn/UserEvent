from flask import Flask, render_template, jsonify, request
import sqlite3
from user_events import event_manager
from lightgcn_data_prep import LightGCNDataPreprocessor
import json
import os
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route('/admin')
def admin_dashboard():
    """관리자 대시보드 페이지"""
    return render_template('admin_dashboard.html')

@app.route('/api/stats')
def get_statistics():
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
        
        return jsonify({
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
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/popular_products')
def get_popular_products():
    """인기 상품 API"""
    try:
        days = request.args.get('days', 7, type=int)
        popular_products = event_manager.get_popular_products(days=days, limit=20)
        return jsonify({'status': 'success', 'data': popular_products})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/user_activity')
def get_user_activity():
    """사용자 활동 추이 API"""
    try:
        days = request.args.get('days', 7, type=int)
        
        conn = sqlite3.connect(event_manager.db_path)
        cursor = conn.cursor()
        
        # 일별 상품 조회 수
        cursor.execute('''
            SELECT DATE(timestamp) as date, COUNT(*) as view_count
            FROM product_views 
            WHERE timestamp >= datetime('now', '-{} days')
            GROUP BY DATE(timestamp)
            ORDER BY date
        '''.format(days))
        
        daily_views = [{'date': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        # 일별 검색 수
        cursor.execute('''
            SELECT DATE(timestamp) as date, COUNT(*) as search_count
            FROM search_events 
            WHERE timestamp >= datetime('now', '-{} days')
            GROUP BY DATE(timestamp)
            ORDER BY date
        '''.format(days))
        
        daily_searches = [{'date': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        # 시간별 활동 (오늘)
        cursor.execute('''
            SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
            FROM product_views 
            WHERE DATE(timestamp) = DATE('now')
            GROUP BY strftime('%H', timestamp)
            ORDER BY hour
        ''')
        
        hourly_activity = [{'hour': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        conn.close()
        
        return jsonify({
            'status': 'success',
            'data': {
                'daily_views': daily_views,
                'daily_searches': daily_searches,
                'hourly_activity': hourly_activity
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/category_stats')
def get_category_stats():
    """카테고리별 통계 API"""
    try:
        days = request.args.get('days', None, type=int)
        
        conn = sqlite3.connect(event_manager.db_path)
        cursor = conn.cursor()
        
        # 기간 필터 조건
        date_filter = ""
        if days:
            date_filter = f"WHERE timestamp >= datetime('now', '-{days} days') AND category != '' AND category IS NOT NULL"
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
        
        return jsonify({'status': 'success', 'data': category_stats})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/reset_statistics', methods=['POST'])
def reset_statistics():
    """통계 데이터 초기화 API"""
    try:
        # 확인을 위한 파라미터
        confirm = request.json.get('confirm', False)
        
        if not confirm:
            return jsonify({'status': 'error', 'message': '확인 파라미터가 필요합니다.'}), 400
        
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
        
        return jsonify({
            'status': 'success', 
            'message': '모든 통계 데이터가 성공적으로 초기화되었습니다.'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/period_stats')
def get_period_statistics():
    """기간별 통계 API"""
    try:
        days = request.args.get('days', 7, type=int)
        
        conn = sqlite3.connect(event_manager.db_path)
        cursor = conn.cursor()
        
        # 기간별 사용자 수
        cursor.execute('''
            SELECT COUNT(DISTINCT user_id) as user_count
            FROM users 
            WHERE created_at >= datetime('now', '-{} days')
        '''.format(days))
        
        period_users = cursor.fetchone()[0]
        
        # 기간별 상품 조회 수
        cursor.execute('''
            SELECT COUNT(*) as view_count
            FROM product_views 
            WHERE timestamp >= datetime('now', '-{} days')
        '''.format(days))
        
        period_views = cursor.fetchone()[0]
        
        # 기간별 검색 수
        cursor.execute('''
            SELECT COUNT(*) as search_count
            FROM search_events 
            WHERE timestamp >= datetime('now', '-{} days')
        '''.format(days))
        
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
        
        return jsonify({
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
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/lightgcn_data')
def get_lightgcn_data():
    """LightGCN 데이터 상태 확인 API"""
    try:
        preprocessor = LightGCNDataPreprocessor()
        data = preprocessor.load_lightgcn_data()
        
        if not data:
            return jsonify({'status': 'error', 'message': 'LightGCN 데이터가 없습니다.'})
        
        return jsonify({
            'status': 'success',
            'data': {
                'n_users': data.get('n_users', 0),
                'n_products': data.get('n_products', 0),
                'n_interactions': data.get('n_interactions', 0),
                'sparsity': data.get('sparsity', 0)
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/regenerate_lightgcn_data')
def regenerate_lightgcn_data():
    """LightGCN 데이터 재생성 API"""
    try:
        min_interactions = request.args.get('min_interactions', 3, type=int)
        
        preprocessor = LightGCNDataPreprocessor()
        data = preprocessor.prepare_lightgcn_data(min_interactions=min_interactions)
        
        if data:
            return jsonify({
                'status': 'success',
                'message': 'LightGCN 데이터가 성공적으로 재생성되었습니다.',
                'data': {
                    'n_users': data['n_users'],
                    'n_products': data['n_products'],
                    'n_interactions': data['n_interactions']
                }
            })
        else:
            return jsonify({'status': 'error', 'message': '데이터가 충분하지 않습니다.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/user_details/<user_id>')
def get_user_details(user_id):
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
            return jsonify({'status': 'error', 'message': '사용자를 찾을 수 없습니다.'}), 404
        
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
        
        return jsonify({
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
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # 환경 변수에서 설정 가져오기
    host = '0.0.0.0'
    port = 7071
    debug = False
    
    print(f"🚀 관리자 대시보드 시작: {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug) 