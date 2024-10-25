from dash import Input, Output, State
import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np

class DashCallbacks:
    """대시보드 콜백 함수들"""
    def setup_callbacks(self, app, data_store):
        """콜백 설정"""
        @app.callback(
            [Output('current-accuracy', 'children'),
             Output('current-domain-loss', 'children'),
             Output('current-mmd-loss', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_metrics(n):
            """현재 메트릭 업데이트"""
            if not data_store['accuracy']:
                return 'N/A', 'N/A', 'N/A'
                
            current_acc = data_store['accuracy'][-1]
            current_domain_loss = data_store['domain_loss'][-1]
            current_mmd_loss = data_store['mmd_loss'][-1]
            
            return (
                f"{current_acc:.2%}",
                f"{current_domain_loss:.4f}",
                f"{current_mmd_loss:.4f}"
            )
            
        @app.callback(
            Output('accuracy-graph', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_accuracy_graph(n):
            """정확도 그래프 업데이트"""
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=list(range(len(data_store['accuracy']))),
                y=data_store['accuracy'],
                mode='lines+markers',
                name='Accuracy'
            ))
            
            fig.update_layout(
                title='Overall Accuracy Trend',
                xaxis_title='Submission',
                yaxis_title='Accuracy',
                yaxis=dict(range=[0, 1])
            )
            
            return fig
            
        @app.callback(
            Output('class-accuracy-graph', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_class_accuracy_graph(n):
            """클래스별 정확도 그래프 업데이트"""
            fig = go.Figure()
            
            for class_id in data_store['class_accuracies']:
                fig.add_trace(go.Scatter(
                    x=list(range(len(data_store['class_accuracies'][class_id]))),
                    y=data_store['class_accuracies'][class_id],
                    mode='lines+markers',
                    name=f'Class {class_id}'
                ))
                
            fig.update_layout(
                title='Class-wise Accuracy Trends',
                xaxis_title='Submission',
                yaxis_title='Accuracy',
                yaxis=dict(range=[0, 1])
            )
            
            return fig
            
        @app.callback(
            Output('domain-loss-graph', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_domain_loss_graph(n):
            """도메인 손실 그래프 업데이트"""
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=list(range(len(data_store['domain_loss']))),
                y=data_store['domain_loss'],
                mode='lines+markers',
                name='Domain Loss'
            ))
            
            fig.add_trace(go.Scatter(
                x=list(range(len(data_store['mmd_loss']))),
                y=data_store['mmd_loss'],
                mode='lines+markers',
                name='MMD Loss'
            ))
            
            fig.update_layout(
                title='Domain Adaptation Loss Trends',
                xaxis_title='Submission',
                yaxis_title='Loss'
            )
            
            return fig
            
        @app.callback(
            Output('confusion-matrix', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_confusion_matrix(n):
            """혼동 행렬 업데이트"""
            if data_store['confusion_matrix'] is None:
                return go.Figure()
                
            labels = [f'Class {i}' for i in range(len(data_store['confusion_matrix']))]
            
            fig = ff.create_annotated_heatmap(
                z=data_store['confusion_matrix'],
                x=labels,
                y=labels,
                colorscale='Blues',
                showscale=True
            )
            
            fig.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted',
                yaxis_title='True',
                xaxis={'side': 'bottom'}
            )
            
            return fig

    @staticmethod
    def create_metric_card(title, value):
        """메트릭 카드 생성"""
        return html.Div([
            html.H4(title),
            html.H2(value)
        ], className='metric-card')