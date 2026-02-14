import pandas as pd
from typing import Dict, List, Optional
from IPython.display import HTML, display


def create_metrics_table(
    metrics_dict: Dict[str, Dict[str, float]],
    metrics_to_show: Optional[List[str]] = None,
    title: str = "Model Comparison",
    highlight_best: bool = True,
    precision: int = 4
) -> HTML:
    if metrics_to_show is None:
        metrics_to_show = [
            'NDCG@100',
            'Recall@100',
            'MRR',
            'Catalog_Coverage',
            'Diversity',
            'Serendipity'
        ]
    
    df = pd.DataFrame(metrics_dict).T
    df = df.fillna(0)
    
    available_metrics = [m for m in metrics_to_show if m in df.columns]
    if not available_metrics:
        raise ValueError(f"None of the requested metrics {metrics_to_show} are available in the data")
    
    df_filtered = df[available_metrics]
    df_formatted = df_filtered.applymap(lambda x: f"{x:.{precision}f}")

    html = f"""
    <style>
        .metrics-table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        .metrics-table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 15px;
            text-align: left;
            font-weight: 600;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metrics-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #f0f0f0;
            font-size: 13px;
        }}
        .metrics-table tr:last-child td {{
            border-bottom: none;
        }}
        .metrics-table tr:hover {{
            background-color: #f8f9ff;
            transition: background-color 0.2s ease;
        }}
        .metrics-table .model-name {{
            font-weight: 600;
            color: #333;
        }}
        .metrics-table .best-value {{
            background-color: #d4edda;
            font-weight: 600;
            color: #155724;
        }}
        .table-title {{
            font-size: 20px;
            font-weight: 700;
            color: #333;
            margin-bottom: 10px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
    </style>
    <div class="table-title">{title}</div>
    <table class="metrics-table">
        <thead>
            <tr>
                <th>Model</th>
    """

    metric_display_names = {
        'NDCG@100': 'NDCG@100',
        'Recall@100': 'Recall@100',
        'MRR': 'MRR',
        'Catalog_Coverage': 'Catalog Coverage',
        'Diversity': 'Diversity',
        'Serendipity': 'Serendipity',
        'NDCG@10': 'NDCG@10',
        'Recall@10': 'Recall@10',
        'MAP@10': 'MAP@10',
        'Precision@10': 'Precision@10'
    }
    
    for metric in available_metrics:
        display_name = metric_display_names.get(metric, metric)
        html += f"                <th>{display_name}</th>\n"
    
    html += """            </tr>
        </thead>
        <tbody>
    """

    best_values = {}
    if highlight_best:
        for metric in available_metrics:
            try:
                best_values[metric] = df_filtered[metric].max()
            except:
                best_values[metric] = None

    for model_name, row in df_formatted.iterrows():
        html += f"            <tr>\n"
        html += f"                <td class='model-name'>{model_name}</td>\n"
        
        for metric in available_metrics:
            value_str = row[metric]

            is_best = False
            if highlight_best and metric in best_values and best_values[metric] is not None:
                try:
                    current_value = df_filtered.loc[model_name, metric]
                    if pd.notna(current_value) and abs(current_value - best_values[metric]) < 1e-10:
                        is_best = True
                except:
                    pass
            
            cell_class = "best-value" if is_best else ""
            html += f"                <td class='{cell_class}'>{value_str}</td>\n"
        
        html += "            </tr>\n"
    
    html += """        </tbody>
    </table>
    """
    
    return HTML(html)


def display_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metrics_to_show: Optional[List[str]] = None,
    title: str = "Model Comparison",
    highlight_best: bool = True,
    precision: int = 4
) -> None:
    html_table = create_metrics_table(
        metrics_dict=metrics_dict,
        metrics_to_show=metrics_to_show,
        title=title,
        highlight_best=highlight_best,
        precision=precision
    )
    display(html_table)


def export_metrics_to_latex(
    metrics_dict: Dict[str, Dict[str, float]],
    metrics_to_show: Optional[List[str]] = None,
    caption: str = "Model Comparison",
    label: str = "tab:model_comparison",
    precision: int = 4
) -> str:
    if metrics_to_show is None:
        metrics_to_show = [
            'NDCG@100',
            'Recall@100',
            'MRR',
            'Catalog_Coverage',
            'Diversity',
            'Serendipity'
        ]
    
    df = pd.DataFrame(metrics_dict).T
    df = df.fillna(0)
    available_metrics = [m for m in metrics_to_show if m in df.columns]
    df_filtered = df[available_metrics]
    metric_latex_names = {
        'NDCG@100': 'NDCG@100',
        'Recall@100': 'Recall@100',
        'MRR': 'MRR',
        'Catalog_Coverage': 'Catalog Coverage',
        'Diversity': 'Diversity',
        'Serendipity': 'Serendipity',
        'NDCG@10': 'NDCG@10',
        'Recall@10': 'Recall@10',
        'MAP@10': 'MAP@10',
        'Precision@10': 'Precision@10'
    }
    
    best_values = {}
    for metric in available_metrics:
        try:
            best_values[metric] = df_filtered[metric].max()
        except:
            best_values[metric] = None

    n_cols = len(available_metrics) + 1
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{{label}}}\n"
    latex += "\\begin{tabular}{l" + "c" * len(available_metrics) + "}\n"
    latex += "\\hline\n"

    latex += "Model"
    for metric in available_metrics:
        display_name = metric_latex_names.get(metric, metric)
        latex += f" & {display_name}"
    latex += " \\\\\n"
    latex += "\\hline\n"

    for model_name, row in df_filtered.iterrows():
        latex += model_name
        
        for metric in available_metrics:
            value = row[metric]
            value_str = f"{value:.{precision}f}"
            is_best = False
            if metric in best_values and best_values[metric] is not None:
                if abs(value - best_values[metric]) < 1e-10:
                    is_best = True
            
            if is_best:
                latex += f" & \\textbf{{{value_str}}}"
            else:
                latex += f" & {value_str}"
        
        latex += " \\\\\n"
    
    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}"
    
    return latex
