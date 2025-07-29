import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from user_events import event_manager
import json
from collections import defaultdict
import pickle
import os

class LightGCNDataPreprocessor:
    def __init__(self, data_dir: str = "lightgcn_data"):
        """LightGCN 데이터 전처리기 초기화"""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def load_user_interaction_data(self) -> pd.DataFrame:
        """사용자-상품 상호작용 데이터 로드"""
        # 이벤트 매니저에서 상호작용 행렬 가져오기
        interaction_data = event_manager.get_user_interaction_matrix()
        
        # DataFrame으로 변환
        rows = []
        for user_id, products in interaction_data['user_product_interactions'].items():
            for product_code, interaction_count in products.items():
                rows.append({
                    'user_id': user_id,
                    'product_code': product_code,
                    'interaction_count': interaction_count
                })
        
        return pd.DataFrame(rows)
    
    def create_user_product_mappings(self, df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
        """사용자 ID와 상품 코드를 정수 인덱스로 매핑"""
        unique_users = df['user_id'].unique()
        unique_products = df['product_code'].unique()
        
        user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        product_code_to_idx = {product_code: idx for idx, product_code in enumerate(unique_products)}
        
        return user_id_to_idx, product_code_to_idx
    
    def prepare_lightgcn_data(self, min_interactions: int = 5) -> Dict[str, Any]:
        """LightGCN용 데이터 준비"""
        print("사용자-상품 상호작용 데이터 로드 중...")
        df = self.load_user_interaction_data()
        
        if df.empty:
            print("상호작용 데이터가 없습니다.")
            return {}
        
        print(f"총 상호작용 수: {len(df)}")
        print(f"고유 사용자 수: {df['user_id'].nunique()}")
        print(f"고유 상품 수: {df['product_code'].nunique()}")
        
        # 최소 상호작용 수 필터링
        user_interaction_counts = df.groupby('user_id')['interaction_count'].sum()
        product_interaction_counts = df.groupby('product_code')['interaction_count'].sum()
        
        valid_users = user_interaction_counts[user_interaction_counts >= min_interactions].index
        valid_products = product_interaction_counts[product_interaction_counts >= min_interactions].index
        
        df_filtered = df[
            (df['user_id'].isin(valid_users)) & 
            (df['product_code'].isin(valid_products))
        ]
        
        print(f"필터링 후 상호작용 수: {len(df_filtered)}")
        print(f"필터링 후 사용자 수: {df_filtered['user_id'].nunique()}")
        print(f"필터링 후 상품 수: {df_filtered['product_code'].nunique()}")
        
        # ID 매핑 생성
        user_id_to_idx, product_code_to_idx = self.create_user_product_mappings(df_filtered)
        
        # 상호작용 행렬 생성
        interaction_matrix = np.zeros((len(user_id_to_idx), len(product_code_to_idx)))
        
        for _, row in df_filtered.iterrows():
            user_idx = user_id_to_idx[row['user_id']]
            product_idx = product_code_to_idx[row['product_code']]
            interaction_matrix[user_idx, product_idx] = row['interaction_count']
        
        # 역매핑 생성
        idx_to_user_id = {idx: user_id for user_id, idx in user_id_to_idx.items()}
        idx_to_product_code = {idx: product_code for product_code, idx in product_code_to_idx.items()}
        
        # 데이터 저장
        data = {
            'interaction_matrix': interaction_matrix,
            'user_id_to_idx': user_id_to_idx,
            'product_code_to_idx': product_code_to_idx,
            'idx_to_user_id': idx_to_user_id,
            'idx_to_product_code': idx_to_product_code,
            'n_users': len(user_id_to_idx),
            'n_products': len(product_code_to_idx),
            'n_interactions': int(interaction_matrix.sum())
        }
        
        # 파일로 저장
        self.save_lightgcn_data(data)
        
        return data
    
    def save_lightgcn_data(self, data: Dict[str, Any]):
        """LightGCN 데이터를 파일로 저장"""
        # numpy 배열 저장
        np.save(os.path.join(self.data_dir, 'interaction_matrix.npy'), data['interaction_matrix'])
        
        # 매핑 정보 저장
        with open(os.path.join(self.data_dir, 'mappings.pkl'), 'wb') as f:
            pickle.dump({
                'user_id_to_idx': data['user_id_to_idx'],
                'product_code_to_idx': data['product_code_to_idx'],
                'idx_to_user_id': data['idx_to_user_id'],
                'idx_to_product_code': data['idx_to_product_code']
            }, f)
        
        # 메타데이터 저장
        with open(os.path.join(self.data_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'n_users': data['n_users'],
                'n_products': data['n_products'],
                'n_interactions': data['n_interactions'],
                'sparsity': 1 - (data['n_interactions'] / (data['n_users'] * data['n_products']))
            }, f, ensure_ascii=False, indent=2)
        
        print(f"LightGCN 데이터가 {self.data_dir} 디렉토리에 저장되었습니다.")
    
    def load_lightgcn_data(self) -> Dict[str, Any]:
        """저장된 LightGCN 데이터 로드"""
        try:
            # numpy 배열 로드
            interaction_matrix = np.load(os.path.join(self.data_dir, 'interaction_matrix.npy'))
            
            # 매핑 정보 로드
            with open(os.path.join(self.data_dir, 'mappings.pkl'), 'rb') as f:
                mappings = pickle.load(f)
            
            # 메타데이터 로드
            with open(os.path.join(self.data_dir, 'metadata.json'), 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            return {
                'interaction_matrix': interaction_matrix,
                'user_id_to_idx': mappings['user_id_to_idx'],
                'product_code_to_idx': mappings['product_code_to_idx'],
                'idx_to_user_id': mappings['idx_to_user_id'],
                'idx_to_product_code': mappings['idx_to_product_code'],
                **metadata
            }
        except FileNotFoundError:
            print("저장된 LightGCN 데이터를 찾을 수 없습니다. 새로 생성합니다.")
            return self.prepare_lightgcn_data()
    
    def get_user_recommendation_candidates(self, user_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """사용자별 추천 후보 상품 가져오기"""
        try:
            data = self.load_lightgcn_data()
            
            if user_id not in data['user_id_to_idx']:
                print(f"사용자 {user_id}의 데이터가 없습니다.")
                return []
            
            user_idx = data['user_id_to_idx'][user_id]
            user_interactions = data['interaction_matrix'][user_idx]
            
            # 이미 상호작용한 상품 제외
            interacted_products = set(np.where(user_interactions > 0)[0])
            all_products = set(range(data['n_products']))
            candidate_products = all_products - interacted_products
            
            # 인기도 기반 추천 (임시)
            product_popularity = data['interaction_matrix'].sum(axis=0)
            candidate_scores = [(idx, product_popularity[idx]) for idx in candidate_products]
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for product_idx, score in candidate_scores[:top_k]:
                product_code = data['idx_to_product_code'][product_idx]
                recommendations.append({
                    'product_code': product_code,
                    'score': float(score),
                    'rank': len(recommendations) + 1
                })
            
            return recommendations
            
        except Exception as e:
            print(f"추천 후보 생성 중 오류: {e}")
            return []
    
    def generate_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """LightGCN 훈련용 데이터 생성"""
        data = self.load_lightgcn_data()
        interaction_matrix = data['interaction_matrix']
        
        # 양의 샘플 (실제 상호작용)
        positive_pairs = np.where(interaction_matrix > 0)
        positive_users = positive_pairs[0]
        positive_products = positive_pairs[1]
        
        # 음의 샘플 (랜덤 샘플링)
        n_negatives = len(positive_users)
        negative_users = np.random.randint(0, data['n_users'], n_negatives)
        negative_products = np.random.randint(0, data['n_products'], n_negatives)
        
        # 음의 샘플에서 실제 상호작용 제거
        for i in range(n_negatives):
            while interaction_matrix[negative_users[i], negative_products[i]] > 0:
                negative_users[i] = np.random.randint(0, data['n_users'])
                negative_products[i] = np.random.randint(0, data['n_products'])
        
        # 훈련 데이터 조합
        train_users = np.concatenate([positive_users, negative_users])
        train_products = np.concatenate([positive_products, negative_products])
        train_labels = np.concatenate([np.ones(n_negatives), np.zeros(n_negatives)])
        
        return train_users, train_products, train_labels

# 사용 예시
if __name__ == "__main__":
    preprocessor = LightGCNDataPreprocessor()
    
    # 데이터 준비
    data = preprocessor.prepare_lightgcn_data(min_interactions=3)
    
    if data:
        print(f"사용자 수: {data['n_users']}")
        print(f"상품 수: {data['n_products']}")
        print(f"상호작용 수: {data['n_interactions']}")
        print(f"희소성: {1 - (data['n_interactions'] / (data['n_users'] * data['n_products'])):.4f}")
        
        # 샘플 사용자 추천 테스트
        if data['n_users'] > 0:
            sample_user = list(data['user_id_to_idx'].keys())[0]
            recommendations = preprocessor.get_user_recommendation_candidates(sample_user, top_k=5)
            print(f"\n사용자 {sample_user}의 추천 후보:")
            for rec in recommendations:
                print(f"  {rec['rank']}. {rec['product_code']} (점수: {rec['score']:.2f})") 