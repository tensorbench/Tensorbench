"""
TensorBench - Compact Dashboard v7 (Custom Profile Toggle)
Fixed profile switching to correctly populate data for LLM/Image modes.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
    QPushButton, QComboBox, QFrame, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QVariantAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QColor, QPainter, QBrush

from tensorbench.core.config import ConfigManager
from tensorbench.core.models import HardwareInfo
from tensorbench.benchmark.runner import BenchmarkRunner, BenchmarkResult
from tensorbench.core.hardware_db import get_gpu_list, get_cpu_list, get_ram_list, find_gpu, find_cpu, find_ram, WORKLOAD_PRESETS
from tensorbench.core.calculator import calculate_sys_score, predict_performance, generate_recommendations
from tensorbench.gui.widgets.performance_map import PerformanceMap

try:
    from tensorbench.hardware.detector import detect_hardware
    DETECTOR_OK = True
except Exception:
    DETECTOR_OK = False


# =============================================================================
# STYLES
# =============================================================================
STYLE = """
QMainWindow, QWidget { background-color: #0f1419; color: #e6e6e6; font-family: "Segoe UI", sans-serif; font-size: 12px; }
QFrame#header_bar { background: #1a1f2e; border-bottom: 1px solid #2a3142; padding: 8px 16px; }
QFrame#card { background: #1a1f2e; border: 1px solid #2a3142; border-radius: 6px; padding: 8px; }
QFrame#config_panel { background: #151922; border: 1px solid #2a3142; border-radius: 6px; padding: 8px; }
QFrame#detected_hw { background: #152015; border: 1px solid #10b981; border-radius: 6px; padding: 8px; }
QComboBox { 
    background: #1a1f2e; color: #e6e6e6; border: 1px solid #2a3142; border-radius: 4px; 
    padding: 4px 8px; min-width: 200px; font-size: 11px;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView { 
    background: #1a1f2e; color: #e6e6e6; selection-background-color: #0078d4;
    selection-color: #ffffff; border: 1px solid #2a3142; font-size: 11px;
}
QPushButton { background: #ff6b00; color: white; border: none; border-radius: 6px; padding: 6px 12px; font-weight: 600; font-size: 11px; }
QPushButton:hover { background: #ff8533; }
QPushButton:disabled { background: #2a3142; color: #5a6270; }
QProgressBar { border: none; border-radius: 3px; background: #2a3142; height: 8px; }
QProgressBar::chunk { background: #ff6b00; border-radius: 3px; }
QLabel#score_label { font-size: 30px; font-weight: 700; font-family: "Consolas", monospace; }
QLabel#tier_label { color: #ff6b00; font-size: 11px; font-weight: bold; }
QLabel#hw_label { color: #10b981; font-size: 10px; font-weight: bold; margin-bottom: 2px; }
QScrollArea { border: none; background: transparent; }
"""


# =============================================================================
# CUSTOM WIDGETS
# =============================================================================

class ProfileToggle(QWidget):
    """Animated sliding pill toggle for LLM / Image profiles"""
    profileChanged = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(34)
        self.setFixedWidth(190)
        self._state = "image"
        self._anim_pos = 4.0

        self._anim = QVariantAnimation(self)
        self._anim.setDuration(250)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._update_pos)

    def _update_pos(self, val):
        self._anim_pos = val
        self.update()

    def mousePressEvent(self, event):
        self.toggle()
        super().mousePressEvent(event)

    def toggle(self):
        self._state = "llm" if self._state == "image" else "image"
        target = 97.0 if self._state == "llm" else 4.0
        self._anim.setStartValue(self._anim_pos)
        self._anim.setEndValue(target)
        self._anim.start()
        self.profileChanged.emit(self._state)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        r = h // 2
        
        pill_w = int(w / 2 - 6)
        pill_h = int(h - 6)
        pill_r = int(r - 3)
        x = int(self._anim_pos)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#151922"))
        painter.drawRoundedRect(0, 0, w, h, r, r)

        painter.setBrush(QColor("#ff6b00"))
        painter.drawRoundedRect(x, 3, pill_w, pill_h, pill_r, pill_r)

        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Medium))
        painter.setPen(QColor("#fff") if self._state == "image" else QColor("#64748b"))
        painter.drawText(0, 0, w//2, h, Qt.AlignmentFlag.AlignCenter, "🎨 Image")
        painter.setPen(QColor("#fff") if self._state == "llm" else QColor("#64748b"))
        painter.drawText(w//2, 0, w//2, h, Qt.AlignmentFlag.AlignCenter, "📝 LLM")
        
        painter.end()


class RightSidebar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        self.detected_panel = QFrame()
        self.detected_panel.setObjectName("detected_hw")
        det_layout = QVBoxLayout(self.detected_panel)
        det_layout.setContentsMargins(6, 6, 6, 6)
        det_layout.setSpacing(2)
        
        det_title = QLabel("✅ DETECTED HARDWARE")
        det_title.setObjectName("hw_label")
        det_layout.addWidget(det_title)
        
        self.lbl_gpu = QLabel("GPU: Scanning...", styleSheet="color: #ccc; font-size: 10px;")
        self.lbl_cpu = QLabel("CPU: Scanning...", styleSheet="color: #ccc; font-size: 10px;")
        self.lbl_ram = QLabel("RAM: Scanning...", styleSheet="color: #ccc; font-size: 10px;")
        for lbl in [self.lbl_gpu, self.lbl_cpu, self.lbl_ram]:
            det_layout.addWidget(lbl)
        layout.addWidget(self.detected_panel)
        
        self.score_gauge = QFrame()
        self.score_gauge.setObjectName("card")
        self.score_gauge.setFixedHeight(75)
        score_layout = QVBoxLayout(self.score_gauge)
        score_layout.setContentsMargins(4, 4, 4, 4)
        score_layout.setSpacing(2)
        score_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.lbl_score = QLabel("0.0", objectName="score_label", styleSheet="color: #ff6b00;", alignment=Qt.AlignmentFlag.AlignCenter)
        self.lbl_tier = QLabel("TIER -", objectName="tier_label", alignment=Qt.AlignmentFlag.AlignCenter)
        self.lbl_delta = QLabel("", styleSheet="color: #10b981; font-size: 9px;", alignment=Qt.AlignmentFlag.AlignCenter)
        for w in [self.lbl_score, self.lbl_tier, self.lbl_delta]:
            score_layout.addWidget(w)
        layout.addWidget(self.score_gauge)
        
        self.config_panel = QFrame()
        self.config_panel.setObjectName("config_panel")
        c_layout = QVBoxLayout(self.config_panel)
        c_layout.setContentsMargins(6, 6, 6, 6)
        c_layout.setSpacing(4)
        
        c_layout.addWidget(QLabel("⚙️ HARDWARE CONFIGURATOR", styleSheet="color: #8892a8; font-size: 9px; font-weight: bold;"))
        for label_text, combo_attr in [("CPU", "combo_cpu"), ("GPU", "combo_gpu"), ("RAM", "combo_ram")]:
            c_layout.addWidget(QLabel(label_text))
            cb = QComboBox()
            cb.addItems(get_cpu_list() if label_text=="CPU" else get_gpu_list() if label_text=="GPU" else get_ram_list())
            cb.setMinimumWidth(200)
            setattr(self, combo_attr, cb)
            c_layout.addWidget(cb)
            
        self.btn_apply = QPushButton("🔄 Recalculate")
        self.btn_apply.setFixedHeight(28)
        c_layout.addWidget(self.btn_apply)
        layout.addWidget(self.config_panel)
        
        self.recs_panel = QFrame()
        self.recs_panel.setObjectName("card")
        r_layout = QVBoxLayout(self.recs_panel)
        r_layout.setContentsMargins(6, 6, 6, 6)
        r_layout.setSpacing(4)
        r_layout.addWidget(QLabel("💡 RECOMMENDATIONS", styleSheet="color: #8892a8; font-size: 9px; font-weight: bold;"))
        self.recs_container = QVBoxLayout()
        self.recs_container.setSpacing(3)
        r_layout.addLayout(self.recs_container)
        layout.addWidget(self.recs_panel)
        layout.addStretch()


# =============================================================================
# MAIN WINDOW
# =============================================================================

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TensorBench")
        self.resize(1280, 750)
        self.setMinimumSize(1050, 650)
        self.setStyleSheet(STYLE)
        
        self.hardware: HardwareInfo | None = None
        self.config = ConfigManager(Path.home() / ".tensorbench")
        self.worker = None
        self.last_result: BenchmarkResult | None = None
        self.active_profile = "image"
        
        self.baseline_config = {"cpu": "", "gpu": "", "ram": ""}
        self.current_config = {"cpu": "", "gpu": "", "ram": ""}

        self._build_ui()
        self._safe_detect()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        header = QFrame()
        header.setObjectName("header_bar")
        hl = QHBoxLayout(header)
        
        lbl_title = QLabel(" TENSOR_BENCH")
        lbl_title.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        lbl_title.setStyleSheet("color: #e6e6e6;")
        hl.addWidget(lbl_title)
        
        hl.addStretch()
        
        self.profile_toggle = ProfileToggle()
        self.profile_toggle.profileChanged.connect(self._on_profile_change)
        hl.addWidget(self.profile_toggle)
        
        hl.addSpacing(20)
        
        self.btn_bench = QPushButton("⚡ INITIATE BENCHMARK")
        self.btn_bench.clicked.connect(self._start_benchmark)
        self.btn_bench.setFixedHeight(32)
        hl.addWidget(self.btn_bench)
        
        main.addWidget(header)

        content = QWidget()
        cl = QHBoxLayout(content)
        cl.setContentsMargins(12, 12, 12, 12)
        cl.setSpacing(12)

        self.perf_map = PerformanceMap()
        cl.addWidget(self.perf_map, 3)

        self.sidebar = RightSidebar()
        self.sidebar.setFixedWidth(270)
        
        self.sidebar.combo_gpu.currentTextChanged.connect(self._on_config_change)
        self.sidebar.combo_cpu.currentTextChanged.connect(self._on_config_change)
        self.sidebar.combo_ram.currentTextChanged.connect(self._on_config_change)
        self.sidebar.btn_apply.clicked.connect(self._apply_config)
        
        cl.addWidget(self.sidebar)
        main.addWidget(content, 1)

    def _on_profile_change(self, profile: str):
        """Handle profile switch"""
        self.active_profile = profile
        
        preset = WORKLOAD_PRESETS.get(profile, WORKLOAD_PRESETS["image"])
        self.perf_map.title.setText(f"{preset['icon']} {preset['name']} Performance")
        
        # ✅ FIXED: Use set_baseline=True to correctly initialize new model set
        self._refresh_dashboard(set_baseline=True)

    def _safe_detect(self):
        if not DETECTOR_OK: return
        try:
            hw = detect_hardware()
            self.hardware = hw
            
            db_gpu = find_gpu(hw.gpu.name)
            db_cpu = find_cpu(hw.cpu.name)
            
            ram_cap = hw.ram.total_gb
            ram_type = hw.ram.type or "DDR4"
            ram_speed = hw.ram.speed_mhz or 3200
            db_ram = find_ram(ram_cap, ram_type, ram_speed)
            
            gpu_display_name = f"{db_gpu.name} ({db_gpu.vram_gb}GB)"
            self.sidebar.lbl_gpu.setText(f"GPU: {gpu_display_name}")
            self.sidebar.lbl_cpu.setText(f"CPU: {hw.cpu.name}")
            self.sidebar.lbl_ram.setText(f"RAM: {db_ram.name}")
            
            self.baseline_config = {
                "gpu": db_gpu.name,
                "cpu": db_cpu.name,
                "ram": db_ram.name
            }
            self.current_config = self.baseline_config.copy()
            
            try: 
                gpu_key = [k for k in get_gpu_list() if db_gpu.name in k][0]
                self.sidebar.combo_gpu.setCurrentText(gpu_key)
            except IndexError:
                pass
            except Exception as e:
                print(f"GPU dropdown error: {e}")
                
            try: self.sidebar.combo_cpu.setCurrentText(db_cpu.name)
            except: pass
            
            try: self.sidebar.combo_ram.setCurrentText(db_ram.name)
            except: pass
            
            self._refresh_dashboard(set_baseline=True)
        except Exception as e:
            print(f"Detection error: {e}")
            import traceback
            traceback.print_exc()
            self.sidebar.lbl_gpu.setText("GPU: Detection failed")

    def _on_config_change(self):
        if not self.baseline_config["gpu"]: return
        new_score = calculate_sys_score(
            self.sidebar.combo_gpu.currentText(),
            self.sidebar.combo_cpu.currentText(),
            self.sidebar.combo_ram.currentText()
        )
        base_score = calculate_sys_score(*self.baseline_config.values())
        delta = new_score.overall - base_score.overall
        
        self.sidebar.lbl_score.setText(f"{new_score.overall:.1f}")
        self.sidebar.lbl_tier.setText(f"TIER {new_score.tier_label}")
        sign = "+" if delta >= 0 else ""
        self.sidebar.lbl_delta.setText(f"{sign}{delta:.1f} vs baseline")
        self.sidebar.lbl_delta.setStyleSheet(f"color: {'#10b981' if delta >= 0 else '#ef4444'}; font-size: 9px;")

    def _apply_config(self):
        self.current_config = {
            "cpu": self.sidebar.combo_cpu.currentText(),
            "gpu": self.sidebar.combo_gpu.currentText(),
            "ram": self.sidebar.combo_ram.currentText()
        }
        self._refresh_dashboard(set_baseline=False)

    def _refresh_dashboard(self, set_baseline: bool = False):
        if not self.hardware: return
        
        gpu_name = self.current_config["gpu"]
        cpu_name = self.current_config["cpu"]
        ram_name = self.current_config["ram"]
        
        gpu_obj = find_gpu(gpu_name)
        gpu_vram = gpu_obj.vram_gb
        
        score = calculate_sys_score(gpu_name, cpu_name, ram_name)
        
        if not set_baseline:
            base_score = calculate_sys_score(*self.baseline_config.values())
            delta = score.overall - base_score.overall
            self.sidebar.lbl_score.setText(f"{score.overall:.1f}")
            self.sidebar.lbl_tier.setText(f"TIER {score.tier_label}")
            sign = "+" if delta >= 0 else ""
            self.sidebar.lbl_delta.setText(f"{sign}{delta:.1f} vs baseline")
            self.sidebar.lbl_delta.setStyleSheet(f"color: {'#10b981' if delta >= 0 else '#ef4444'}; font-size: 9px;")
        
        predictions = predict_performance(gpu_name, ram_name, self.active_profile)
        unit = WORKLOAD_PRESETS[self.active_profile]["unit"]
        
        if set_baseline:
            self.perf_map.set_baseline(predictions, gpu_vram, unit)
        else:
            self.perf_map.update_predictions(predictions, gpu_vram, unit)
        
        recs = generate_recommendations(gpu_name, cpu_name, ram_name, self.active_profile)
        while self.sidebar.recs_container.count():
            self.sidebar.recs_container.takeAt(0).widget().deleteLater()
        
        for rec in recs:
            lbl = QLabel(rec)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("background: #151922; border-left: 2px solid #ff6b00; padding: 4px 6px; border-radius: 4px; color: #ddd; font-size: 10px;")
            self.sidebar.recs_container.addWidget(lbl)

    def _start_benchmark(self):
        if not self.hardware:
            return QMessageBox.warning(self, "Внимание", "Сначала дождитесь сканирования.")
        self.btn_bench.setEnabled(False)
        self.btn_bench.setText("⏳ Testing...")
        
        class BW(QThread):
            progress = pyqtSignal(float, str)
            finished = pyqtSignal(object)
            error = pyqtSignal(str)
            def __init__(self, hw): super().__init__(); self.hw=hw; self.runner=BenchmarkRunner()
            def run(self):
                try:
                    import asyncio
                    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
                    self.runner.set_progress_callback(lambda v,t: self.progress.emit(v,t))
                    res = loop.run_until_complete(self.runner.run_async(self.hw))
                    loop.close(); self.finished.emit(res)
                except Exception as e: self.error.emit(str(e))

        self.worker = BW(self.hardware)
        self.worker.finished.connect(self._on_done)
        self.worker.error.connect(self._on_err)
        self.worker.start()

    def _on_done(self, res: BenchmarkResult):
        self.last_result = res
        self.btn_bench.setEnabled(True)
        self.btn_bench.setText("⚡ INITIATE BENCHMARK")
        self._refresh_dashboard(set_baseline=False)

    def _on_err(self, msg: str):
        self.btn_bench.setEnabled(True)
        self.btn_bench.setText("⚡ INITIATE BENCHMARK")
        if "llama-cpp" in msg.lower():
            QMessageBox.warning(self, "Зависимость", "pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121")
        else:
            QMessageBox.critical(self, "Ошибка", msg)

    def closeEvent(self, event):
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.worker.terminate()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 11))
    w = App()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()