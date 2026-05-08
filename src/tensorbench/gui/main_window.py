"""
TensorBench - Professional Benchmark UI (PyQt6 Strict Typing Fixed)
"""
from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
    QPushButton, QProgressBar, QRadioButton, QButtonGroup, QFrame, QMessageBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QLinearGradient, QPainter, QBrush, QPen

from tensorbench.core.config import ConfigManager
from tensorbench.core.models import ScenarioType, HardwareInfo
from tensorbench.benchmark.runner import BenchmarkRunner, BenchmarkResult

try:
    from tensorbench.hardware.detector import detect_hardware
    DETECTOR_OK = True
except Exception:
    DETECTOR_OK = False


# =============================================================================
# STYLES
# =============================================================================
STYLE = """
QMainWindow, QWidget { background-color: #0f1419; color: #e6e6e6; font-family: "Segoe UI", sans-serif; }
QFrame#card { background: #1a1f2e; border: 1px solid #2a3142; border-radius: 12px; padding: 16px; }
QFrame#upgrade_card { background: #1a2332; border: 1px solid #2a3142; border-radius: 8px; padding: 16px; margin-bottom: 12px; }
QFrame#upgrade_card:hover { border-color: #0078d4; background: #1f2a3d; }
QTabWidget::pane { border: none; background: #151922; border-radius: 12px; }
QTabBar::tab { background: #1a1f2e; color: #8892a8; padding: 10px 20px; border-top-left-radius: 8px; border-top-right-radius: 8px; margin-right: 4px; font-weight: 600; }
QTabBar::tab:selected { background: #0078d4; color: white; }
QTableWidget { background: #151922; border: 1px solid #2a3142; border-radius: 8px; gridline-color: #2a3142; }
QTableWidget::item { padding: 8px; }
QHeaderView::section { background: #1a1f2e; color: #8892a8; padding: 8px; border: none; font-weight: bold; }
QPushButton { background: #0078d4; color: white; border: none; border-radius: 8px; padding: 12px 24px; font-weight: 600; }
QPushButton:hover { background: #1084d8; }
QPushButton:disabled { background: #2a3142; color: #5a6270; }
QProgressBar { border: none; border-radius: 4px; background: #2a3142; height: 10px; }
QProgressBar::chunk { background: #0078d4; border-radius: 4px; }
QLabel#metric { font-size: 24px; font-weight: 700; font-family: "Consolas", monospace; }
"""


# =============================================================================
# CUSTOM WIDGETS
# =============================================================================

class RankBar(QWidget):
    def __init__(self, percentile: int, parent=None):
        super().__init__(parent)
        self.percentile = max(1, min(99, percentile))
        self.setFixedHeight(60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        pos = (self.percentile / 100) * (w - 40) + 20

        # Background track
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#2a3142")))
        painter.drawRoundedRect(20, 25, w - 40, 10, 5, 5)

        # Gradient fill
        grad = QLinearGradient(20, 0, w - 20, 0)
        grad.setColorAt(0, QColor("#ef4444"))      # ✅ Fixed: wrapped in QColor()
        grad.setColorAt(0.4, QColor("#f59e0b"))    # ✅ Fixed
        grad.setColorAt(1, QColor("#10b981"))      # ✅ Fixed
        painter.setBrush(QBrush(grad))
        painter.drawRoundedRect(20, 25, w - 40, 10, 5, 5)

        # Marker
        painter.setBrush(QBrush(QColor("#fff")))
        painter.setPen(QPen(QColor("#fff"), 2))
        painter.drawEllipse(int(pos) - 6, 20, 12, 20)

        # Text
        painter.setPen(QColor("#e6e6e6"))
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"Вы быстрее {self.percentile}% пользователей")


class UpgradeCard(QFrame):
    def __init__(self, title: str, impact: str, details: list[str], color: str = "#0078d4"):
        super().__init__()
        self.setObjectName("upgrade_card")
        self.setStyleSheet(f"""#upgrade_card {{ border-left: 4px solid {color}; }}""")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        
        head = QHBoxLayout()
        lbl_title = QLabel(title)
        lbl_title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        lbl_title.setStyleSheet(f"color: {color};")
        head.addWidget(lbl_title)
        
        lbl_impact = QLabel(impact)
        lbl_impact.setStyleSheet("color: #8892a8; font-size: 12px; background: #0f1419; padding: 4px 8px; border-radius: 4px;")
        head.addWidget(lbl_impact)
        head.addStretch()
        lay.addLayout(head)
        
        for d in details:
            lay.addWidget(QLabel(f"• {d}", styleSheet="color: #b0b8c8; font-size: 12px; padding-left: 8px;"))


# =============================================================================
# MAIN WINDOW
# =============================================================================
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TensorBench")
        self.resize(1100, 750)
        self.setStyleSheet(STYLE)
        
        self.hardware: HardwareInfo | None = None
        self.config = ConfigManager(Path.home() / ".tensorbench")
        self.worker = None
        self.last_result: BenchmarkResult | None = None

        self._build_ui()
        self._safe_detect()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)
        main.setContentsMargins(24, 24, 24, 24)
        main.setSpacing(20)

        head = QHBoxLayout()
        head.addWidget(QLabel("<span style='font-size:24px;font-weight:700;color:#fff;'>TensorBench</span><br><span style='color:#8892a8;'>AI Hardware Benchmark</span>"))
        self.badge = QFrame()
        self.badge.setStyleSheet("background: #1a1f2e; border: 1px solid #2a3142; border-radius: 12px; padding: 12px;")
        badge_lay = QHBoxLayout(self.badge)
        self.sys_lbl = QLabel(" Сканирование...", styleSheet="color:#8892a8;")
        badge_lay.addWidget(self.sys_lbl)
        head.addWidget(self.badge, 1)
        main.addLayout(head)

        self.prof_card = QFrame()
        self.prof_card.setStyleSheet("background: #1a1f2e; border: 1px solid #2a3142; border-radius: 12px; padding: 16px;")
        prof_lay = QHBoxLayout(self.prof_card)
        self.prof_group = QButtonGroup()
        for i, (k, t) in enumerate([("llm","📝 LLM"), ("image","🎨 Картинки"), ("audio","🎙️ Аудио"), ("video","🎥 Видео")]):
            rb = QRadioButton(t)
            rb.setChecked(i==0)
            rb.setStyleSheet("QRadioButton { color: #b0b8c8; padding: 8px 12px; }")
            prof_lay.addWidget(rb)
            self.prof_group.addButton(rb, i)
            setattr(self, f"rb_{k}", rb)
        main.addWidget(self.prof_card)

        self.btn = QPushButton("🚀 Запустить бенчмарк")
        self.btn.setFixedHeight(50)
        self.btn.clicked.connect(self._start)
        main.addWidget(self.btn)

        self.prog = QProgressBar()
        self.prog.setVisible(False)
        self.prog.setFixedHeight(8)
        main.addWidget(self.prog)

        self.tabs = QTabWidget()
        self.tabs.setVisible(False)
        self.tabs.setDocumentMode(True)
        
        self.tab_result = QWidget()
        self.tab_res_lay = QVBoxLayout(self.tab_result)
        self.rank_bar_placeholder = QWidget()
        self.rank_bar_placeholder.setLayout(QVBoxLayout())
        self.rank_bar_placeholder.layout().setContentsMargins(0,0,0,0)
        self.tab_res_lay.addWidget(self.rank_bar_placeholder)
        
        self.table_models = QTableWidget()
        self.table_models.setColumnCount(4)
        self.table_models.setHorizontalHeaderLabels(["Модель", "Размер", "Прогноз TPS", "Статус"])
        self.table_models.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table_models.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table_models.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table_models.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.tab_res_lay.addWidget(self.table_models)
        self.tabs.addTab(self.tab_result, "🏆 Ваш Результат")

        self.tab_upgrade = QWidget()
        self.tab_upg_lay = QVBoxLayout(self.tab_upgrade)
        self.upg_container = QWidget()
        self.upg_container.setLayout(QVBoxLayout())
        self.upg_container.layout().setContentsMargins(0,0,0,0)
        self.tab_upg_lay.addWidget(self.upg_container)
        self.tabs.addTab(self.tab_upgrade, "🚀 Симулятор Апгрейда")

        self.tab_health = QWidget()
        self.tab_hlt_lay = QVBoxLayout(self.tab_health)
        self.health_container = QWidget()
        self.health_container.setLayout(QVBoxLayout())
        self.health_container.layout().setContentsMargins(0,0,0,0)
        self.tab_hlt_lay.addWidget(self.health_container)
        self.tabs.addTab(self.tab_health, "📊 Анализ Системы")

        main.addWidget(self.tabs, 1)

    def _refresh_ui(self):
        if not self.last_result: return
        self._render_tab_1()
        self._render_tab_2()
        self._render_tab_3()

    def _render_tab_1(self):
        while self.rank_bar_placeholder.layout().count():
            self.rank_bar_placeholder.layout().takeAt(0).widget().deleteLater()
            
        tps = self.last_result.tps
        vram = self.hardware.gpu.vram_total_gb or 0
        score = 0
        if tps > 30: score += 40
        elif tps > 15: score += 20
        if vram >= 16: score += 40
        elif vram >= 8: score += 20
        percentile = min(99, max(5, score + 15))
        
        self.rank_bar_placeholder.layout().addWidget(RankBar(percentile))
        
        color = "#10b981" if percentile > 75 else "#f59e0b"
        status = "Высокий" if percentile > 75 else "Средний"
        lbl = QLabel(f"Базовая скорость: <b>{tps:.0f} TPS</b> | Ваш класс: <span style='color:{color}'>{status}</span>")
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setStyleSheet("color: #8892a8; padding: 10px 0; font-size: 14px;")
        self.rank_bar_placeholder.layout().addWidget(lbl)
        
        self.table_models.setRowCount(0)
        models = [
            ("Qwen2.5-0.5B", "0.5B", tps, True),
            ("Mistral-7B", "7B", tps * 0.07, vram >= 4),
            ("Llama-3-8B", "8B", tps * 0.06, vram >= 5),
            ("Qwen2.5-14B", "14B", tps * 0.035, vram >= 8),
            ("Command-R-35B", "35B", tps * 0.015, vram >= 16),
        ]
        
        for name, size, pred_tps, fits in models:
            row = self.table_models.rowCount()
            self.table_models.insertRow(row)
            self.table_models.setItem(row, 0, QTableWidgetItem(name))
            self.table_models.setItem(row, 1, QTableWidgetItem(size))
            
            tps_val = QTableWidgetItem(f"{pred_tps:.1f}")
            tps_val.setForeground(QColor("#10b981" if pred_tps >= 10 else "#f59e0b" if pred_tps >= 3 else "#ef4444"))
            tps_val.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
            self.table_models.setItem(row, 2, tps_val)
            
            self.table_models.setItem(row, 3, QTableWidgetItem("✅ GPU" if fits else "⚠️ Гибрид" if pred_tps > 1 else "❌ CPU"))
        self.table_models.resizeColumnsToContents()

    def _render_tab_2(self):
        while self.upg_container.layout().count():
            self.upg_container.layout().takeAt(0).widget().deleteLater()
            
        vram = self.hardware.gpu.vram_total_gb or 0
        ram = self.hardware.ram.total_gb or 0
        ram_type = self.hardware.ram.type or ""
        gpu_name = self.hardware.gpu.name or ""
        tps = self.last_result.tps
        
        gpu_tier = 1.0
        if "4090" in gpu_name: gpu_tier = 3.0
        elif "4080" in gpu_name: gpu_tier = 2.5
        elif "4070" in gpu_name: gpu_tier = 1.8
        elif "3090" in gpu_name: gpu_tier = 2.2
        elif "3080" in gpu_name: gpu_tier = 2.0
        
        details_gpu = [
            f"Текущий TPS (7B): {tps * 0.07:.1f}",
            f"Прогноз с новой картой: {(tps * 0.07) * (2.5/max(gpu_tier, 0.1)):.1f} TPS",
            f"Прирост: +{(2.5/max(gpu_tier, 0.1) - 1)*100:.0f}%" if gpu_tier < 2.5 else "Топ уровень"
        ]
        if gpu_tier < 2.0:
            self.upg_container.layout().addWidget(UpgradeCard("🎮 Апгрейд Видеокарты (до RTX 4070 Ti)", "Высокий эффект", details_gpu, "#0078d4"))

        if ram < 32:
            self.upg_container.layout().addWidget(UpgradeCard(
                "💾 Добавить Оперативную Память (до 32GB/64GB)", 
                "Открывает доступ к тяжелым моделям", 
                [f"Текущий лимит: ~{int(vram/0.7)}B", "С апгрейдом: Запуск 30B+ через оффлоад", "Устранит своп на диск"], 
                "#f59e0b"
            ))

        if "DDR4" in ram_type:
             self.upg_container.layout().addWidget(UpgradeCard(
                "⚡ Смена платформы на DDR5 / PCIe 4.0", "Средний эффект", 
                ["Ускорение гибридного режима", "Прирост +15-20% в CPU-оффлоаде"], "#8b5cf6"))
        
        if self.upg_container.layout().count() == 0:
            self.upg_container.layout().addWidget(QLabel("✅ Ваша система оптимальна. Апгрейд не требуется.", styleSheet="color:#10b981; padding:20px; font-size:14px;"))

    def _render_tab_3(self):
        while self.health_container.layout().count():
            self.health_container.layout().takeAt(0).widget().deleteLater()
            
        vram = self.hardware.gpu.vram_total_gb or 0
        ram = self.hardware.ram.total_gb or 0
        ram_type = self.hardware.ram.type or "DDR"
        ram_speed = self.hardware.ram.speed_mhz or 0
        
        self.health_container.layout().addWidget(QLabel("🎮 Видеопамять (VRAM)", styleSheet="color:#00d4ff; font-weight:bold; padding-bottom:5px;"))
        pb_vram = QProgressBar()
        pb_vram.setValue(int((vram / 24) * 100))
        pb_vram.setFormat(f"{vram} GB / 24 GB (Эталон)")
        pb_vram.setStyleSheet("QProgressBar::chunk { background: #00d4ff; }")
        self.health_container.layout().addWidget(pb_vram)
        self.health_container.layout().addSpacing(20)

        self.health_container.layout().addWidget(QLabel("🧠 Оперативная память (RAM)", styleSheet="color:#f59e0b; font-weight:bold; padding-bottom:5px;"))
        pb_ram = QProgressBar()
        pb_ram.setValue(int((ram / 64) * 100))
        pb_ram.setFormat(f"{ram} GB {ram_type}-{ram_speed}")
        pb_ram.setStyleSheet("QProgressBar::chunk { background: #f59e0b; }")
        self.health_container.layout().addWidget(pb_ram)
        self.health_container.layout().addSpacing(20)

        bottleneck = "Система сбалансирована."
        if vram < 8: bottleneck = "🔴 Критическое узкое место: Мало видеопамяти (VRAM)."
        elif ram < 16: bottleneck = "🟠 Ограничение: Мало оперативной памяти."
        elif "GTX" in (self.hardware.gpu.name or ""): bottleneck = "🟡 Ограничение: Старая архитектура GPU."
        
        lbl = QLabel(f"💡 <b>Вердикт:</b> {bottleneck}")
        lbl.setWordWrap(True)
        lbl.setStyleSheet("background: #1a1f2e; padding: 16px; border-radius: 8px; color: #e6e6e6; font-size: 14px; border: 1px solid #2a3142;")
        self.health_container.layout().addWidget(lbl)

    def _safe_detect(self):
        if not DETECTOR_OK: return self._on_hw_fail("Detector missing")
        class DT(QThread):
            result = pyqtSignal(object); error = pyqtSignal(str)
            def run(self):
                try: self.result.emit(detect_hardware())
                except Exception as e: self.error.emit(str(e))
        self.dt = DT()
        self.dt.result.connect(self._on_hw_ready)
        self.dt.error.connect(self._on_hw_fail)
        self.dt.start()
        if not self.dt.wait(5000):
            self.dt.terminate()
            self._on_hw_fail("Timeout")

    def _on_hw_ready(self, hw):
        self.hardware = hw
        self.sys_lbl.setText(f"💻 {hw.cpu.name[:25]}... | 🎮 {hw.gpu.name} ({hw.gpu.vram_total_gb:.0f}GB) |  {hw.ram.total_gb:.0f}GB")
        try: self.config.scenarios
        except: pass

    def _on_hw_fail(self, msg):
        self.sys_lbl.setText("❌ Ошибка определения.")
        QMessageBox.warning(self, "Внимание", f"Железо: {msg}")

    def _start(self):
        if not self.hardware: return QMessageBox.warning(self, "Внимание", "Дождитесь сканирования.")
        self.btn.setEnabled(False); self.btn.setText("⏳ Тест...")
        self.prog.setVisible(True); self.prog.setValue(0)
        self.tabs.setVisible(False)

        class BW(QThread):
            progress = pyqtSignal(float, str); finished = pyqtSignal(object); error = pyqtSignal(str)
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
        self.worker.progress.connect(lambda v, _: self.prog.setValue(int(v*100)))
        self.worker.finished.connect(self._on_done)
        self.worker.error.connect(self._on_err)
        self.worker.start()

    def _on_done(self, res):
        self.btn.setEnabled(True); self.btn.setText("🚀 Запустить бенчмарк")
        self.prog.setVisible(False)
        self.last_result = res
        self.tabs.setVisible(True)
        self._refresh_ui()
        self.tabs.setCurrentIndex(0)

    def _on_err(self, msg):
        self.btn.setEnabled(True); self.btn.setText("🚀 Запустить бенчмарк")
        self.prog.setVisible(False)
        if "llama-cpp" in msg.lower():
            QMessageBox.warning(self, "Зависимость", "pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121")
        else:
            QMessageBox.critical(self, "Ошибка", msg)

    def closeEvent(self, event):
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning(): self.worker.terminate()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 11))
    w = App()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()