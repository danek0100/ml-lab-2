ML-Lab-2<br>
<hr>
Цель обучения: предсказание цены дома.
<hr>

## Лучшие результаты моделей
CatBoost
<ul>
    <li>R2:  0.91614 [exp-dee1e]</li>
    <li>MSE:  5.9234e+08 [exp-7e9ea]</li>
    <li>MAPE:  0.08925 [exp-de10a]</li>
    <li>EV: 0.91732 [exp-7e9ea]</li>
</ul>

<hr>
XGBoost
<ul>
    <li>R2:  0.8933 [main]</li>
    <li>MSE:  8.5795e+088 [main]</li>
    <li>MAPE:  0.094424 [exp-dee1e]</li>
    <li>EV: 0.89377 [main]</li>
</ul>
        
<hr>

# Обоснования выбора метрик
Метрики – R2, MSE, MAPE, Explained variance

Описание метрик:

R2 - это соотношение между тем, насколько хороша наша модель, и тем, насколько хороша модель наивного среднего.
MSE - метрика, которая в основном измеряет среднеквадратичную ошибку наших прогнозов. Для каждой точки вычисляется квадратная разница между прогнозами и целью, а затем усредняются эти значения.
Для полной оценки модели так же используем метрики:

MAPE - коэффициент, не имеющий размерности, с очень простой интерпретацией, в нашем случае измерение происходит в долях.
The Explained variance score - похожа на оценку R2 с заметной разницей в том, что она не учитывает систематические смещения в прогнозе.
Целевая метрика: <b>R2</b>

Данная метрика легко масштабируется и интерпретируется, что позволяет быстро определить насколько точно модель предсказывает цену дома.
<hr>

## DVC

Организация pipeline (dvc_dag.txt):<br>
<pre>
       +-----------------+          
       | data_processing |          
       +-----------------+          
                 *                  
                 *                  
                 *                  
      +--------------------+        
      | feature_generation |        
      +--------------------+        
                 *                  
                 *                  
                 *                  
            +-------+               
            | train |               
            +-------+*              
           **         **            
         **             **          
        *                 *         
+----------+         +-----------+  
| evaluate |         | inference |  
+----------+         +-----------+  
+--------------+ 
| dynamic_plot | 
+--------------+ 
</pre>
<br>
Хранение данных на GoogleDrive: https://drive.google.com/drive/folders/1QV2eoGSkIbj50g7k1Uy8kHRv7eEfW_5X<br>
Работа с метриками: dvc_metrics_show.txt<br>
Работа с экспериментами:
<ul>
    <li>Для эффективного и массового проведения экспериментов написан скрипт <b>src/exp_generate.py</b>, в котором можно задать все нужные параметры для проверки моделей;</li> 
    <li>Алгоритм работы <b>КРАЙНЕ НЕЭФФЕКТИВЕН</b> и представляет собой полный перебор все возможных вариаций подборов параметров, однако можно назвать красиво и сказать, что это <b>Grid Search</b> в рамках DVC экспериментов  :-)</li> 
    <li>Конечно, существует DVCLive, но судя по описанию он подходит для отслеживания обучения модели в рамках одного набора параметров и визуализации шагов обучения; </li>
    <li>Данный скрипт делает схожую работу только в рамках различных параметров, сохраняя результаты всех проведённых эксперементов, что позволяет не только использовать dvc exp show, но и визуализировать метрики, находя лучшие решения;</li>
    <li>Для красивой визуализации используется скрипт: dvc_exp_show.bat - удаляющий лишнее параметры;</li>
    <li>Пример вывода результатов для <b>128+</b> экспериментов: <b>exp_show.txt</b>;</li>
</ul>

<br>
Работа с графиками (dvc_plots): <br>
Пример вывода статистики обучения от CatBoost: 

![alt text](https://github.com/danek0100/ml-lab-2/blob/main/dvc_plots/visualization_learn_error_catboost_1000_iterations.png?raw=true)
<br><br>
Пример вывода метрики r2 от XGBoost модели для 4-х экспериментов:

![alt text](https://github.com/danek0100/ml-lab-2/blob/main/dvc_plots/visualization_dynamic_xgboost_4_exp.png?raw=true)
<hr>

Project Organization<br>
------------

Была добавлена папка src/inference - для запуска модели на обычных данных. <br>
Были добавлены некоторые скрипты для эффективной работы <br>

------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py
    │   │
    │   └── inference  <- Scripts to work with generated models
    │       └── inference.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
