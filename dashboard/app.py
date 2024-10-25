import dash
from dash import dcc, html
import plotly.graph_objs as go
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import queue
from .callbacks import DashCallbacks

class MonitoringDashboard:
    """실시간 모니터링 대시보드"""
    def __init__(self, config_path: str = 'configs/config.yaml'):
        # 대시보드 초기화
        self.app = dash.Dash(__name__)
        self.callbacks = DashCallbacks()
        
        # 데이터 저장소
        self.data_store = {
            'accuracy': [],
            'class_accuracies': {i: [] for i in range(5)},
            'domain_loss': [],
            'mmd_loss': [],
            'timestamps': [],
            'confusion_matrix': None
        }
        
        # 결과 큐
        self.results_queue = queue.Queue()
        
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """대시보드 레이아웃 설정"""
        self.app.layout = html.Div([
            # 헤더
            html.H1("Ship Classification Monitoring Dashboard",
                   style={'textAlign': 'center', 'marginBottom': 30}),
            
            # 실시간 메트릭 카드
            html.Div([
                html.Div([
                    html.H3("Current Accuracy"),
                    html.H4(id='current-accuracy', children='N/A')
                ], className='metric-card'),
                html.Div([
                    html.H3("Domain Loss"),
                    html.H4(id='current-domain-loss', children='N/A')
                ], className='metric-card'),
                html.Div([
                    html.H3("MMD Loss"),
                    html.H4(id='current-mmd-loss', children='N/A')
                ], className='metric-card')
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': 30}),
            
            # 그래프 섹션
            html.Div([
                # 정확도 추이
                html.Div([
                    html.H3("Accuracy Trend"),
                    dcc.Graph(id='accuracy-graph')
                ], className='graph-container'),
                
                # 클래스별 정확도
                html.Div([
                    html.H3("Class-wise Accuracy"),
                    dcc.Graph(id='class-accuracy-graph')
                ], className='graph-container'),
                
                # 도메인 손실
                html.Div([
                    html.H3("Domain Loss Trend"),
                    dcc.Graph(id='domain-loss-graph')
                ], className='graph-container'),
                
                # 혼동 행렬
                html.Div([
                    html.H3("Confusion Matrix"),
                    dcc.Graph(id='confusion-matrix')
                ], className='graph-container')
            ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'}),
            
            # 업데이트 간격 설정
            dcc.Interval(
                id='interval-component',
                interval=1*1000,  # 1초마다 업데이트
                n_intervals=0
            ),
            
            # 데이터 저장소
            dcc.Store(id='data-store')
        ])
        
    def setup_callbacks(self):
        """콜백 함수 설정"""
        self.callbacks.setup_callbacks(self.app, self.data_store)
        
    def update_data(self, results: dict):
        """새로운 결과로 데이터 업데이트"""
        timestamp = datetime.now()
        
        self.data_store['accuracy'].append(results['accuracy'])
        self.data_store['timestamps'].append(timestamp)
        
        for class_id, acc in results['class_accuracies'].items():
            self.data_store['class_accuracies'][class_id].append(acc)
            
        self.data_store['domain_loss'].append(results['domain_loss'])
        self.data_store['mmd_loss'].append(results['mmd_loss'])
        
        if 'confusion_matrix' in results:
            self.data_store['confusion_matrix'] = results['confusion_matrix']
            
    def run(self, host='localhost', port=8050, debug=False):
        """대시보드 실행"""
        self.app.run_server(host=host, port=port, debug=debug)

# CSS 스타일
app_css = '''
.metric-card {
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    background-color: white;
    text-align: center;
    width: 250px;
}

.graph-container {
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    background-color: white;
    margin-bottom: 20px;
}
'''

if __name__ == '__main__':
    dashboard = MonitoringDashboard()
    dashboard.run()