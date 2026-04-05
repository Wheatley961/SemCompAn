import streamlit as st
import pandas as pd
import stanza
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Set
from collections import defaultdict, Counter
import io

# ============================================================================
# КОНФИГУРАЦИЯ СТРАНИЦЫ
# ============================================================================
st.set_page_config(
    page_title="Программа автоматического компонентного анализа LexiSeme",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# КЛАССЫ ОБРАБОТКИ
# ============================================================================

class LinguisticProcessor:
    """Класс для обработки естественного языка на базе Stanza"""
    
    def __init__(self):
        self.nlp = None
        self.is_loaded = False
    
    def load_model(self):
        """Загрузка модели Stanza"""
        if not self.is_loaded:
            with st.spinner('🔄 Загрузка лингвистической модели...'):
                try:
                    self.nlp = stanza.Pipeline('ru', verbose=False, process_mwt=True)
                    self.is_loaded = True
                    st.success('✅ Модель русского языка загружена')
                except Exception as e:
                    st.error(f'❌ Ошибка загрузки модели: {e}')
                    st.info('Попробуйте выполнить: python -m stanza.download ru')
                    return False
        return True
    
    def extract_semes(self, definition: str) -> List[str]:
        """Выделение сем из дефиниции"""
        if not self.nlp:
            return []
        
        doc = self.nlp(definition)
        semes = []
        
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.upos is None:
                    continue
                
                is_candidate = word.upos in ["NOUN", "ADJ"]
                
                if word.upos == "VERB":
                    if word.feats and ('VerbForm=Part' in word.feats or 'PartType' in word.feats):
                        is_candidate = True
                
                if is_candidate:
                    lemma = word.lemma.lower()
                    stop_words = ['этот', 'тот', 'какой', 'который', 'свой', 'весь', 'сам', 'такой']
                    if len(lemma) > 2 and lemma not in stop_words:
                        semes.append(lemma)
        
        return list(set(semes))


class ComponentialAnalysisSystem:
    """Система компонентного анализа"""
    
    def __init__(self, processor: LinguisticProcessor):
        self.processor = processor
        self.dictionary_name: str = ""
        self.lexical_entries: Dict[str, str] = {}
        self.matrix_data = None
        self.all_semes: Set[str] = set()
    
    def add_entry(self, word: str, definition: str):
        """Добавление записи"""
        if word and definition:
            self.lexical_entries[word] = definition
    
    def remove_entry(self, word: str):
        """Удаление записи"""
        if word in self.lexical_entries:
            del self.lexical_entries[word]
    
    def clear_all(self):
        """Очистка всех данных"""
        self.lexical_entries = {}
        self.dictionary_name = ""
        self.matrix_data = None
        self.all_semes = set()
    
    def perform_analysis(self) -> bool:
        """Выполнение анализа"""
        if not self.lexical_entries:
            return False
        
        all_semes = set()
        word_semes_map = {}
        
        for word, definition in self.lexical_entries.items():
            semes = self.processor.extract_semes(definition)
            word_semes_map[word] = semes
            all_semes.update(semes)
        
        if not all_semes:
            return False
        
        self.all_semes = all_semes
        sorted_semes = sorted(list(all_semes))
        sorted_words = sorted(list(self.lexical_entries.keys()))
        
        matrix_data = []
        for word in sorted_words:
            row_data = {'СЛОВО': word}
            word_specific_semes = set(word_semes_map[word])
            for seme in sorted_semes:
                row_data[seme] = 1 if seme in word_specific_semes else 0
            matrix_data.append(row_data)
        
        self.matrix_data = pd.DataFrame(matrix_data)
        self.matrix_data.set_index('СЛОВО', inplace=True)
        
        return True
    
    def get_statistics(self) -> Dict:
        """Получение статистики анализа"""
        if self.matrix_data is None:
            return {}
        
        total_words = len(self.matrix_data)
        total_semes = len(self.matrix_data.columns)
        
        # Интегральные семы (присутствуют у >50% слов)
        integral_semes = []
        differential_semes = []
        
        for col in self.matrix_data.columns:
            count = self.matrix_data[col].sum()
            percentage = (count / total_words) * 100
            if percentage >= 50:
                integral_semes.append((col, count, percentage))
            else:
                differential_semes.append((col, count, percentage))
        
        integral_semes.sort(key=lambda x: x[2], reverse=True)
        differential_semes.sort(key=lambda x: x[2], reverse=True)
        
        return {
            'total_words': total_words,
            'total_semes': total_semes,
            'integral_semes': integral_semes[:10],
            'differential_semes': differential_semes[:10],
            'density': (self.matrix_data.sum().sum() / (total_words * total_semes)) * 100 if total_semes > 0 else 0
        }
    
    def export_to_csv(self) -> bytes:
        """Экспорт в CSV"""
        if self.matrix_data is None:
            return b""
        
        csv_buffer = io.StringIO()
        self.matrix_data.to_csv(csv_buffer)
        return csv_buffer.getvalue().encode('utf-8')
    
    def export_to_excel(self) -> bytes:
        """Экспорт в Excel"""
        if self.matrix_data is None:
            return b""
        
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            self.matrix_data.to_excel(writer, sheet_name='Матрица сем')
        return excel_buffer.getvalue()


# ============================================================================
# ИНТЕРФЕЙС STREAMLIT
# ============================================================================

def main():
    # Стилизация
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
        }
        .stDataFrame {
            font-size: 0.9rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Заголовок
    st.markdown('<div class="main-header">Программа автоматического компонентного анализа лексического значения LexiSeme</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Система семного анализа на базе Stanza</div>', unsafe_allow_html=True)
    
    # Инициализация сессии
    if 'processor' not in st.session_state:
        st.session_state.processor = LinguisticProcessor()
    
    if 'analysis_system' not in st.session_state:
        st.session_state.analysis_system = ComponentialAnalysisSystem(st.session_state.processor)
    
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Боковая панель
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # Загрузка модели
        st.subheader("🔧 Модель NLP")
        if st.button("Загрузить модель Stanza", key="load_model"):
            st.session_state.processor.load_model()
        
        if st.session_state.processor.is_loaded:
            st.success("✅ Модель активна")
        else:
            st.warning("⚠️ Модель не загружена")
        
        st.divider()
        
        # Управление данными
        st.subheader("📁 Управление данными")
        
        if st.button("🗑️ Очистить все данные", key="clear_all"):
            st.session_state.analysis_system.clear_all()
            st.session_state.data_loaded = False
            st.rerun()
        
        # Экспорт
        st.divider()
        st.subheader("💾 Экспорт результатов")
        
        if st.session_state.data_loaded and st.session_state.analysis_system.matrix_data is not None:
            csv_data = st.session_state.analysis_system.export_to_csv()
            st.download_button(
                label="📥 Скачать CSV",
                data=csv_data,
                file_name="componential_analysis.csv",
                mime="text/csv"
            )
            
            excel_data = st.session_state.analysis_system.export_to_excel()
            st.download_button(
                label="📥 Скачать Excel",
                data=excel_data,
                file_name="componential_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        st.divider()
        
        # Информация
        st.subheader("ℹ️ О системе")
        st.info("""
        **Компонентный анализ** — метод семантического анализа, 
        раскладывающий значение слова на минимальные смысловые компоненты (семы).
        
        **Интегральные семы** — общие для группы слов.
        
        **Дифференциальные семы** — уникальные признаки.
        """)
    
    # Основной интерфейс
    if not st.session_state.processor.is_loaded:
        st.warning("⚠️ Сначала загрузите модель NLP в боковой панели!")
        return
    
    # Вкладка 1: Ввод данных
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Ввод данных", "📊 Матрица анализа", "📈 Визуализация", "📋 Статистика"])
    
    with tab1:
        st.header("Ввод лексических данных")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Название словаря
            dict_name = st.text_input(
                "📖 Название словаря-источника",
                value=st.session_state.analysis_system.dictionary_name,
                key="dict_name_input"
            )
            st.session_state.analysis_system.dictionary_name = dict_name
            
            st.divider()
            
            # Добавление новых записей
            st.subheader("Добавить новое слово")
            col_word, col_def, col_btn = st.columns([2, 5, 1])
            
            with col_word:
                new_word = st.text_input("Слово", key="new_word")
            
            with col_def:
                new_definition = st.text_input("Дефиниция", key="new_def")
            
            with col_btn:
                st.write("")  # Отступ
                st.write("")  # Отступ
                if st.button("➕ Добавить", key="add_entry"):
                    if new_word and new_definition:
                        st.session_state.analysis_system.add_entry(new_word, new_definition)
                        st.session_state.data_loaded = False
                        st.success(f"✅ Слово '{new_word}' добавлено!")
                    else:
                        st.error("❌ Заполните оба поля!")
        
        with col2:
            # Список добавленных слов
            st.subheader("📋 Добавленные слова")
            if st.session_state.analysis_system.lexical_entries:
                for word in st.session_state.analysis_system.lexical_entries.keys():
                    col_del1, col_del2 = st.columns([4, 1])
                    with col_del1:
                        st.text(word)
                    with col_del2:
                        if st.button("🗑️", key=f"del_{word}"):
                            st.session_state.analysis_system.remove_entry(word)
                            st.session_state.data_loaded = False
                            st.rerun()
            else:
                st.info("Нет добавленных слов")
        
        st.divider()
        
        # Быстрое добавление примеров
        st.subheader("🚀 Быстрые примеры")
        if st.button("Загрузить пример 'Мебель'", key="load_example"):
            examples = {
                "Стул": "Предмет мебели в виде широкой доски на ножках для сидения со спинкой",
                "Табурет": "Предмет мебели в виде скамейки без спинки для сидения",
                "Кресло": "Мебель для сидения одного человека с мягким сиденьем и подлокотниками",
                "Диван": "Мебель для сидения и лежания нескольких человек с мягким наполнением",
                "Стол": "Предмет мебели в виде широкой доски на ножках для работы или еды"
            }
            for word, defn in examples.items():
                st.session_state.analysis_system.add_entry(word, defn)
            st.session_state.data_loaded = False
            st.success("✅ Примеры загружены!")
            st.rerun()
        
        # Кнопка анализа
        st.divider()
        col_analyze1, col_analyze2, col_analyze3 = st.columns([2, 1, 2])
        with col_analyze2:
            if st.button("🔍 Выполнить анализ", key="perform_analysis", type="primary", use_container_width=True):
                if len(st.session_state.analysis_system.lexical_entries) < 2:
                    st.error("❌ Добавьте минимум 2 слова для анализа!")
                else:
                    with st.spinner('🔄 Выполняется компонентный анализ...'):
                        success = st.session_state.analysis_system.perform_analysis()
                        if success:
                            st.session_state.data_loaded = True
                            st.success("✅ Анализ успешно выполнен!")
                        else:
                            st.error("❌ Не удалось выделить семы из дефиниций!")
    
    with tab2:
        st.header("Матрица семного анализа")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Сначала выполните анализ во вкладке 'Ввод данных'!")
        else:
            system = st.session_state.analysis_system
            
            # Фильтры
            col_filter1, col_filter2 = st.columns(2)
            with col_filter1:
                min_semes = st.slider("Минимум слов с семой", 1, len(system.lexical_entries), 1)
            
            # Отображение матрицы
            st.subheader("📊 Бинарная матрица присутствия сем")
            
            # Цветовая схема
            styled_matrix = system.matrix_data.copy()
            
            def color_matrix(val):
                if val == 1:
                    return 'background-color: #2ecc71; color: white; font-weight: bold;'
                else:
                    return 'background-color: #ecf0f1; color: #7f8c8d;'
            
            st.dataframe(
                styled_matrix.style.map(color_matrix),
                use_container_width=True,
                height=400
            )
            
            # Детали по словам
            st.divider()
            st.subheader("📖 Детали дефиниций")
            
            for word, definition in system.lexical_entries.items():
                with st.expander(f"📌 {word}"):
                    st.write(f"**Дефиниция:** {definition}")
                    semes = system.processor.extract_semes(definition)
                    st.write(f"**Выделенные семы:** {', '.join(semes)}")
    
    with tab3:
        st.header("Визуализация результатов")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Сначала выполните анализ!")
        else:
            system = st.session_state.analysis_system
            stats = system.get_statistics()
            
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                st.subheader("📊 Распределение сем по словам")
                
                # Тепловая карта
                fig_heatmap = px.imshow(
                    system.matrix_data.values,
                    labels=dict(x="Семы", y="Слова", color="Наличие"),
                    x=system.matrix_data.columns,
                    y=system.matrix_data.index,
                    color_continuous_scale=[[0, '#ecf0f1'], [1, '#2ecc71']],
                    aspect="auto"
                )
                fig_heatmap.update_layout(height=400, xaxis_title="Семы", yaxis_title="Слова")
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            with col_viz2:
                st.subheader("📈 Плотность сем")
                
                # Диаграмма плотности
                seme_counts = system.matrix_data.sum().sort_values(ascending=True)
                
                fig_bar = px.bar(
                    x=seme_counts.values,
                    y=seme_counts.index,
                    orientation='h',
                    labels={'x': 'Количество слов', 'y': 'Сема'},
                    color=seme_counts.values,
                    color_continuous_scale='Viridis'
                )
                fig_bar.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            st.divider()
            
            # Круговая диаграмма
            col_viz3, col_viz4 = st.columns(2)
            
            with col_viz3:
                st.subheader("🔵 Интегральные и дифференциальные семы")
                
                integral_count = len(stats['integral_semes'])
                differential_count = len(stats['differential_semes'])
                
                fig_pie = px.pie(
                    values=[integral_count, differential_count],
                    names=['Интегральные', 'Дифференциальные'],
                    title='Типы сем',
                    color_discrete_sequence=['#3498db', '#e74c3c']
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_viz4:
                st.subheader("📊 Топ-10 наиболее частотных сем")
                
                top_semes = stats['integral_semes'][:10]
                if top_semes:
                    fig_top = px.bar(
                        x=[s[0] for s in top_semes],
                        y=[s[1] for s in top_semes],
                        labels={'x': 'Сема', 'y': 'Количество слов'},
                        color=[s[2] for s in top_semes],
                        color_continuous_scale='Blues'
                    )
                    fig_top.update_layout(showlegend=False)
                    st.plotly_chart(fig_top, use_container_width=True)
                else:
                    st.info("Нет данных для отображения")
    
    with tab4:
        st.header("Статистика анализа")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Сначала выполните анализ!")
        else:
            system = st.session_state.analysis_system
            stats = system.get_statistics()
            
            # Метрики
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                st.metric("Всего слов", stats['total_words'])
            
            with col_m2:
                st.metric("Всего сем", stats['total_semes'])
            
            with col_m3:
                st.metric("Плотность матрицы", f"{stats['density']:.1f}%")
            
            with col_m4:
                st.metric("Словарь", system.dictionary_name or "Не указан")
            
            st.divider()
            
            # Интегральные семы
            col_int, col_diff = st.columns(2)
            
            with col_int:
                st.subheader("🔵 Интегральные семы (≥50% слов)")
                if stats['integral_semes']:
                    df_integral = pd.DataFrame(
                        stats['integral_semes'],
                        columns=['Сема', 'Количество', 'Процент']
                    )
                    df_integral['Процент'] = df_integral['Процент'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(df_integral, use_container_width=True)
                else:
                    st.info("Нет интегральных сем")
            
            with col_diff:
                st.subheader("🔴 Дифференциальные семы (<50% слов)")
                if stats['differential_semes']:
                    df_diff = pd.DataFrame(
                        stats['differential_semes'],
                        columns=['Сема', 'Количество', 'Процент']
                    )
                    df_diff['Процент'] = df_diff['Процент'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(df_diff, use_container_width=True)
                else:
                    st.info("Нет дифференциальных сем")
            
            st.divider()
            
            # Интерпретация
            st.subheader("Рекомендации по интерпретации")
            
            st.info("""
            **Как работать с результатами:**
            
            1. **Интегральные семы** показывают родовые признаки (например, 'предмет', 'мебель')
            2. **Дифференциальные семы** выявляют видовые отличия (например, 'спинка', 'подлокотник')
            3. **Плотность матрицы** показывает насколько слова семантически близки
            4. **Пустые ячейки** означают отсутствие признака в дефиниции
            """)


if __name__ == "__main__":
    main()
