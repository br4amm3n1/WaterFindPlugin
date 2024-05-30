import os.path
import datetime
from typing import List, Tuple, Optional
import processing
import time

from dataclasses import dataclass
from qgis.PyQt.QtCore import pyqtSignal

from PyQt5 import uic
from PyQt5.QtGui import QColor

from qgis.PyQt.QtWidgets import (QFileDialog, QLabel, QVBoxLayout, QPushButton,
                                 QMessageBox, QGroupBox, QDialog)

from qgis.core import (QgsProject, QgsVectorLayer, QgsRasterLayer,
                       QgsTask, QgsApplication, QgsMessageLog, Qgis,
                       QgsRendererCategory, QgsSymbol, QgsCategorizedSymbolRenderer)

from .utils.split_geotiff import split_geotiff

from .segment_model.segment_water_model import SegmentationModel, AbstractModel

FORM_CLASS, _ = uic.loadUiType(os.path.join(os.path.dirname(__file__), 'model_settings_dialog.ui'))


class ModelSettingsDialog(QDialog, FORM_CLASS):
    def __init__(self, iface, parent: object = None) -> None:
        super().__init__(parent)
        self.__iface = iface
        self.setupUi(self)

        self.__save_path: str = os.path.join(os.path.dirname(__file__), "storage", "vector_masks")
        self.__path_to_model: Optional[str] = None

        self.__image_paths: List[str] = []

        self.init_ui()

    def init_ui(self) -> None:
        self.connect_slots()

        self.setFixedSize(491, 416)
        self.setWindowTitle("Модуль распознавания ДЗЗ")

        # self.progressBar.setVisible(False)
        self.stopProcessBtn.setEnabled(False)

        self.lineEditSavePath.setReadOnly(True)
        self.lineEditSelectedImages.setReadOnly(True)
        self.lineEditPathToModel.setReadOnly(True)

        text = os.path.join(os.path.dirname(__file__), "storage", "vector_masks")
        self.lineEditSavePath.setText(f"{text}")

        self.lineEditSelectedImages.setText("0")

    def connect_slots(self):
        self.getPathsImagesBtn.clicked.connect(self.get_images_paths)
        self.getResultBtn.clicked.connect(self.run_model)
        self.getImageForSplitBtn.clicked.connect(self.get_tiles)
        self.stopProcessBtn.clicked.connect(self.stop_model)
        self.setPathToVectorBtn.clicked.connect(self.set_save_path)
        self.setPathToModelBtn.clicked.connect(self.set_path_to_model)

    def set_save_path(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            None,
            "Выберите каталог для сохранения",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.ReadOnly
        )

        if directory:
            self.__save_path = directory
            self.lineEditSavePath.setText(f"{self.__save_path}")

    def set_path_to_model(self) -> None:
        path = QFileDialog.getOpenFileName(
            self,
            caption="Выберите модель сегментации",
            filter="ONNX files (*.onnx);;All Files (*)"
        )[0]
        dot_index = path.strip().rfind('.')

        if path:
            if path[dot_index:].lower() == ".onnx":
                self.__path_to_model = path
                self.lineEditPathToModel.setText(f"{self.__path_to_model}")

    def get_images_paths(self) -> None:
        self.__image_paths = QFileDialog.getOpenFileNames(self, caption="Выберите изображения")[0]

        self.lineEditSelectedImages.setText(f"{len(self.__image_paths)}")

    def get_tiles(self) -> None:
        image_path = QFileDialog.getOpenFileName(self, caption="Выберите изображение")[0]
        dot_index = image_path.strip().rfind('.')

        output_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "storage", "tiles_storage"))

        if image_path[dot_index:].lower() == ".tif":
            split_geotiff(image_path, output_folder, 512)

    def run_model(self) -> None:
        if self.__path_to_model:
            if self.__image_paths:
                model = SegmentationModel()
                model.path = self.__path_to_model
                model.images = self.__image_paths

                task = GettingMaskTask(self.__iface)
                task.model = model
                task.output_path_polygonized = self.__save_path
                task.task_finished.connect(self.__finished)
                task.remaining_time_changed.connect(self.__remaining_time_changed)
                # task.progress_changed.connect(self.__progress_changed)

                QgsApplication.taskManager().addTask(task)
                QgsApplication.processEvents()

                self.stopProcessBtn.setEnabled(True)
                self.getResultBtn.setEnabled(False)
                self.getPathsImagesBtn.setEnabled(False)
                self.setPathToVectorBtn.setEnabled(False)
            else:
                QMessageBox().warning(
                    self,
                    "Предупреждение",
                    "Выберите изображения, чтобы начать сегментацию",
                    buttons=QMessageBox.Ok
                )
        else:
            QMessageBox().warning(
                self,
                "Предупреждение",
                "Не выбрана модель сегментации изображений",
                buttons=QMessageBox.Ok
            )

    def stop_model(self) -> None:
        QgsApplication.taskManager().cancelAll()
        self.stopProcessBtn.setEnabled(False)

    def __finished(self, result: bool) -> None:
        self.stopProcessBtn.setEnabled(False)
        self.getResultBtn.setEnabled(True)
        self.getPathsImagesBtn.setEnabled(True)
        self.setPathToVectorBtn.setEnabled(True)

        if result:
            QgsMessageLog.logMessage('Task "Getting masks" was completed successfully', "GettingMaskTask", Qgis.Info)

            QMessageBox().information(
                self,
                "Успех",
                "Сегментация изображения успешно завершена.",
                buttons=QMessageBox.Ok
            )
        else:
            QgsMessageLog.logMessage(
                'Task "Getting masks" was completed not successfully',
                "GettingMaskTask",
                Qgis.Info
            )

    def __remaining_time_changed(self, remaining_time):
        minutes, seconds = divmod(remaining_time, 60)
        hours, minutes = divmod(minutes, 60)

        timee = [hours, minutes, seconds]
        for i in range(len(timee)):
            timee[i] = ModelSettingsDialog.format_time(timee[i])

        self.labelRemainingTime.setText("До завершения осталось: " + timee[0] + " : " + timee[1] + " : " + timee[2])

    # def __progress_changed(self, progress_value):
    #     self.progressBar.setValue(progress_value)

    @staticmethod
    def format_time(time):
        if time == time % 10:
            return "0" + f"{time:.0f}"
        else:
            return f"{time:.0f}"


@dataclass
class Entities:
    """
    Уникальные значения, получаемые в результате
    работы модели сегментации
    """

    WATER = 255


class GettingMaskTask(QgsTask):
    __stopProcess: bool = False
    task_finished = pyqtSignal(bool)
    remaining_time_changed = pyqtSignal(float)
    # progress_changed = pyqtSignal(float)

    def __init__(self, iface, parent: object = None) -> None:
        super().__init__(parent, QgsTask.AllFlags)

        self.__iface = iface

        self.__output_path_merged = os.path.join(os.path.dirname(__file__), "storage", "pred_masks")
        self.__output_path_polygonized = os.path.join(os.path.dirname(__file__), "storage", "vector_masks")
        self.__masks = []

        self.__model = None

    @property
    def output_path_polygonized(self):
        return self.__output_path_polygonized

    @output_path_polygonized.setter
    def output_path_polygonized(self, path):
        self.__output_path_polygonized = path

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model: AbstractModel) -> None:
        self.__model = model

    def run(self) -> bool:
        QgsMessageLog.logMessage('Started task "Getting masks"', "GettingMaskTask", Qgis.Info)
        mask_generator = self.__model.get_result()

        self.setProgress(0)

        counter = 0
        total_count = len(self.__model.images)

        start_time = time.time()

        while True:
            try:
                self.__check_interrupt()

                original_image, mask = next(mask_generator)
                counter += 1

                path_to_image_mask = os.path.abspath(os.path.join(
                    os.path.dirname(__file__),
                    "storage",
                    "pred_masks",
                    os.path.basename(original_image)
                ))

                self.__model.save_mask(mask, path_to_image_mask, original_image)
                self.__masks.append(path_to_image_mask)

                progress = ((counter / total_count) * 80)
                # self.progress_changed.emit(progress)
                self.setProgress(progress)

                elapsed_time = time.time() - start_time
                remaining_time = (elapsed_time / progress) * (80 - progress)

                self.remaining_time_changed.emit(remaining_time)

            except StopIteration:
                self.__check_interrupt()

                merge_result = self.__merge(self.__masks)

                # self.progress_changed.emit(90)
                self.setProgress(90)

                if merge_result[0]:
                    polygonize_result = self.__polygonize(merge_result[1])

                    # self.progress_changed.emit(99)
                    self.setProgress(99)

                    if polygonize_result[0]:
                        vlayer = QgsVectorLayer(
                            polygonize_result[1],
                            f"{GettingMaskTask.get_filename(polygonize_result[1])}",
                            "ogr"
                        )

                        GettingMaskTask.__categorize(vlayer)

                        QgsProject.instance().addMapLayer(vlayer)

                        # self.progress_changed.emit(100)
                        self.setProgress(100)
                    else:
                        # self.progress_changed.emit(0)
                        self.setProgress(0)
                        self.task_finished.emit(False)
                        return False
                else:
                    # self.progress_changed.emit(0)
                    self.setProgress(0)
                    self.task_finished.emit(False)
                    return False

                # self.progress_changed.emit(0)
                self.setProgress(100)
                break
            except Exception:
                # self.progress_changed.emit(0)
                self.setProgress(0)
                self.task_finished.emit(False)
                return False

        self.task_finished.emit(True)
        return True

    def cancel(self) -> None:
        QgsMessageLog.logMessage('Task "Getting masks" was canceled', "GettingMaskTask", Qgis.Info)
        super().cancel()
        self.__stopProcess = True

    def __check_interrupt(self) -> None:
        if self.__stopProcess:
            self.remaining_time_changed.emit(0)
            raise Exception("Работа прервана пользователем")

    def __merge(self, masks: List[str]) -> Tuple[bool, str]:
        raster_images = []

        output_path = os.path.abspath(os.path.join(
            self.__output_path_merged,
            f"merged_{datetime.datetime.now():%Y%m%d_%H%M%S}" + ".tif")
        )

        for mask in masks:
            raster_images.append(QgsRasterLayer(mask, "temp_raster"))

        try:
            processing.run("gdal:merge",
                           {
                               "INPUT": raster_images,
                               "OUTPUT": output_path
                           })
            return True, output_path
        except Exception as e:
            return False, str(e)

    def __polygonize(self, path: str) -> Tuple[bool, str]:
        raster = QgsRasterLayer(path, "raster")

        output_path = os.path.abspath(os.path.join(
            self.__output_path_polygonized,
            f"polygonized_{datetime.datetime.now():%Y%m%d_%H%M%S}" + ".gpkg")
        )

        try:
            processing.run("gdal:polygonize",
                           {
                               'INPUT': raster,
                               'BAND': 1,
                               'FIELD': 'DN',
                               'EIGHT_CONNECTEDNESS': False,
                               'EXTRA': '',
                               'OUTPUT': output_path
                           })
            return True, output_path
        except Exception as e:
            return False, str(e)

    @staticmethod
    def __categorize(vlayer: QgsVectorLayer) -> None:
        field_name = "DN"

        categories = []

        index_field = vlayer.fields().indexFromName(field_name)
        unique_values = vlayer.uniqueValues(index_field)

        for value in unique_values:
            if value == Entities.WATER:
                symbol = QgsSymbol.defaultSymbol(2)
                symbol.setColor(QColor(128, 184, 220, 255))
                symbol.setOpacity(0.7)
            else:
                continue

            category = QgsRendererCategory()
            category.setValue(value)
            category.setSymbol(symbol)
            category.setLabel(str(value))

            categories.append(category)

        renderer = QgsCategorizedSymbolRenderer(field_name, categories)
        vlayer.setRenderer(renderer)

    @staticmethod
    def get_filename(path: str) -> str:
        basename = os.path.basename(path)
        dot_index = basename.strip().rfind('.')

        return basename[:dot_index]
