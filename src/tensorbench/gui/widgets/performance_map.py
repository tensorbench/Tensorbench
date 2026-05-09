"""
Performance Map - Multi-Profile Support
Fixed syntax typo in update_chart signature.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea, 
    QGridLayout, QProgressBar
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

try:
    import pyqtgraph as pg
    HAS_GRAPH = True
except ImportError:
    HAS_GRAPH = False

from tensorbench.core.calculator import PerformancePrediction
from tensorbench.core.hardware_db import WORKLOAD_PRESETS

class ModelCard(QFrame):
    def __init__(self, model_name: str, speed: float, vram_gb: float, gpu_vram: int, unit: str = "it/s", parent=None):
        super().__init__(parent)
        fits_vram = vram_gb <= gpu_vram if gpu_vram > 0 else False
        
        self.setStyleSheet("""
            QFrame {
                background: #1e293b; 
                border: 1px solid #334155;
                border-radius: 12px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # 1. Header
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        
        name_lbl = QLabel(model_name.split(" (")[0])
        name_lbl.setStyleSheet("color: #f8fafc; font-size: 14px; font-weight: 600; background: transparent; border: none; padding: 0;")
        header.addWidget(name_lbl)
        header.addStretch()
        
        vram_lbl = QLabel(f"{vram_gb:.1f} GB")
        vram_color = "#ef4444" if not fits_vram else "#10b981" 
        vram_bg = "#450a0a" if not fits_vram else "#064e3b"
        
        vram_lbl.setStyleSheet(f"""
            background: {vram_bg};
            color: {vram_color};
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: bold;
            font-family: 'Consolas';
            border: none;
        """)
        header.addWidget(vram_lbl)
        layout.addLayout(header)

        # 2. Speed (Hero Number)
        speed_layout = QHBoxLayout()
        speed_layout.setContentsMargins(0, 0, 0, 0)
        speed_layout.setSpacing(6)
        
        speed_val = QLabel(f"{speed:.1f}")
        speed_val.setStyleSheet("color: #f8fafc; font-size: 36px; font-weight: 800; font-family: 'Consolas'; background: transparent; border: none; padding: 0;")
        speed_layout.addWidget(speed_val)
        
        speed_unit = QLabel(unit)
        speed_unit.setStyleSheet("color: #94a3b8; font-size: 14px; background: transparent; border: none; padding: 0; margin-bottom: 4px;")
        speed_layout.addWidget(speed_unit)
        
        speed_layout.addStretch()
        layout.addLayout(speed_layout)

        # 3. Progress Bar
        ref_max = 50.0
        pct = min(100, int((speed / ref_max) * 100))
        bar_color = "#10b981" if speed >= ref_max * 0.2 else "#f59e0b" if speed >= ref_max * 0.06 else "#ef4444"
        
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(pct)
        bar.setTextVisible(False)
        bar.setStyleSheet(f"""
            QProgressBar {{ background: #0f172a; border: none; border-radius: 4px; height: 8px; }}
            QProgressBar::chunk {{ background: {bar_color}; border-radius: 4px; }}
        """)
        layout.addWidget(bar)


class ComparisonChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(10)
        
        title = QLabel("Performance Comparison vs Baseline")
        title.setStyleSheet("color: #94a3b8; font-size: 12px; font-weight: 600; padding-left: 4px;")
        layout.addWidget(title)

        legend_html = """
        <div style="background: #0f172a; padding: 6px 12px; border-radius: 6px; display: flex; gap: 16px; font-size: 11px; color: #94a3b8; width: fit-content;">
            <span>● <b style="color:#475569">Baseline</b></span>
            <span>● <b style="color:#f97316">Upgraded (Δ %)</b></span>
        </div>
        """
        legend_lbl = QLabel(legend_html)
        layout.addWidget(legend_lbl)

        if HAS_GRAPH:
            self.plot = pg.PlotWidget(background="#0f172a")
            self.plot.showGrid(x=True, y=True, alpha=0.1)
            self.plot.hideAxis('left')
            axis = self.plot.getAxis('bottom')
            axis.setStyle(tickTextOffset=8)
            layout.addWidget(self.plot)
        else:
            lbl = QLabel("PyQtGraph not installed.")
            lbl.setStyleSheet("color: #64748b; padding: 20px;")
            layout.addWidget(lbl)

    # ✅ FIXED: Removed space typo in argument name
    def update_chart(self, baseline_data: dict, new_data: dict):
        if not HAS_GRAPH: return
        self.plot.clear()
        
        base_vals = list(baseline_data.values())
        new_vals = list(new_data.values())
        x = list(range(len(base_vals)))
        
        b1 = pg.BarGraphItem(x=x, height=base_vals, width=0.35, brush='#475569')
        b2 = pg.BarGraphItem(x=[i + 0.4 for i in x], height=new_vals, width=0.35, brush='#f97316')
        self.plot.addItem(b1)
        self.plot.addItem(b2)
        
        max_val = max(max(base_vals), max(new_vals), 1)
        LABEL_OFFSET = max(3.0, max_val * 0.3)
        
        for i, (bv, nv) in enumerate(zip(base_vals, new_vals)):
            if bv > 0:
                t1 = pg.TextItem(text=f"{bv:.1f}", color='#cbd5e1', anchor=(0.5, 0))
                t1.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
                t1.setPos(i, bv + LABEL_OFFSET)
                self.plot.addItem(t1)
                
            if nv > 0 and bv > 0:
                pct = ((nv - bv) / bv) * 100
                sign = "+" if pct >= 0 else ""
                color = '#10b981' if pct >= 0 else '#ef4444'
                t2 = pg.TextItem(text=f"{sign}{pct:.0f}%", color=color, anchor=(0.5, 0))
                t2.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
                t2.setPos(i + 0.4, nv + LABEL_OFFSET)
                self.plot.addItem(t2)
            elif nv > 0:
                t2 = pg.TextItem(text=f"{nv:.1f}", color='#f97316', anchor=(0.5, 0))
                t2.setFont(QFont("Consolas", 10))
                t2.setPos(i + 0.4, nv + LABEL_OFFSET)
                self.plot.addItem(t2)
        
        self.plot.setYRange(0, max_val + LABEL_OFFSET + 2.0)


class PerformanceMap(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("performance_map")
        self.setStyleSheet("#performance_map { background: #0f172a; border-radius: 12px; padding: 20px; border: 1px solid #1e293b; }")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(16)

        header = QHBoxLayout()
        icon = QLabel("📊")
        icon.setStyleSheet("font-size: 18px;")
        header.addWidget(icon)
        
        self.title = QLabel("Model Performance Grid")
        self.title.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        self.title.setStyleSheet("color: #f8fafc; padding-left: 6px;")
        header.addWidget(self.title)
        
        header.addStretch()
        self.subtitle = QLabel("Baseline: Current Hardware")
        self.subtitle.setStyleSheet("color: #64748b; font-size: 11px;")
        header.addWidget(self.subtitle)
        main_layout.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        grid_container = QWidget()
        self.grid_layout = QGridLayout(grid_container)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(16)
        scroll.setWidget(grid_container)
        main_layout.addWidget(scroll)

        self.chart = ComparisonChart()
        self.chart.setFixedHeight(220) 
        main_layout.addWidget(self.chart)

        self.baseline_preds = {}
        self.gpu_vram = 16
        self.current_unit = "it/s"

    # Stores full prediction objects to access VRAM later
    def set_baseline(self, predictions: list[PerformancePrediction], gpu_vram: int = 16, unit: str = "it/s"):
        self.baseline_preds = {p.model_name: p for p in predictions}
        if gpu_vram > 0: self.gpu_vram = gpu_vram
        self.current_unit = unit
        self.subtitle.setText("Baseline: Current Hardware")
        self._render(self.baseline_preds, self.baseline_preds)

    # Stores full prediction objects and extracts speeds for chart
    def update_predictions(self, predictions: list[PerformancePrediction], gpu_vram: int = None, unit: str = None):
        if gpu_vram is not None and gpu_vram > 0:
            self.gpu_vram = gpu_vram
        if unit:
            self.current_unit = unit
            
        new_preds = {p.model_name: p for p in predictions}
        
        if not self.baseline_preds:
            self.set_baseline(predictions, gpu_vram, unit)
            return
        
        self.subtitle.setText("🔄 Comparing vs Baseline")
        self._render(self.baseline_preds, new_preds)
        
        # Extract speeds for chart
        base_speeds = {name: p.predicted_speed for name, p in self.baseline_preds.items()}
        new_speeds = {name: p.predicted_speed for name, p in new_preds.items()}
        self.chart.update_chart(base_speeds, new_speeds)

    def _render(self, baseline: dict, new: dict):
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        models = list(baseline.keys())
        row, col = 0, 0
        for model in models:
            base_pred = baseline[model] # Now an object
            new_pred = new.get(model)
            
            if not new_pred: continue
            
            speed = new_pred.predicted_speed
            vram = base_pred.vram_usage_gb # Access VRAM from object
            
            card = ModelCard(model, speed, vram, self.gpu_vram, self.current_unit)
            self.grid_layout.addWidget(card, row, col)
            col += 1
            if col >= 3: 
                col = 0
                row += 1