import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from datetime import datetime
import struct


class BMWCANAnalyzer:
    def __init__(self, candump_file_path):
        """
        BMW CAN 데이터 분석기 초기화

        Args:
            candump_file_path (str): CAN dump 파일 경로
        """
        self.file_path = candump_file_path
        self.raw_data = []
        self.parsed_data = []
        self.df = None

        # BMW CAN ID 분류 (추정값 - 실제 차량별로 다를 수 있음)
        # 주의: 이 값들은 일반적인 패턴 기반 추정값이며, 실제 BMW 차량의
        # 정확한 CAN ID는 해당 모델의 DBC 파일이나 공식 문서에서 확인해야 합니다.
        self.bmw_can_categories = {
            'Engine_Estimated': [0x0A5, 0x0CA, 0x0C4, 0x0C0, 0x03C, 0x12F],
            'Transmission_Estimated': [0x1A1, 0x1C5, 0x1E4],
            'Body_Electronics_Estimated': [0x175, 0x173, 0x199, 0x19A, 0x197, 0x0D9],
            'Comfort_Estimated': [0x33E, 0x3B8, 0x3B9, 0x3FD, 0x3F9],
            'Safety_Systems_Estimated': [0x2EB, 0x265, 0x301, 0x314, 0x316],
            'Infotainment_Estimated': [0x1F8, 0x1BA, 0x360, 0x0A2],
            'Climate_Estimated': [0x163, 0x368, 0x2F4],
            'Lighting_Estimated': [0x104, 0x297, 0x304],
            'Unknown': []
        }

    def load_and_parse_data(self):
        """CAN dump 파일을 로드하고 파싱"""
        print("CAN dump 파일 로딩 중...")

        can_pattern = r'\((\d+\.\d+)\)\s+(\w+)\s+([0-9A-F]+)#([0-9A-F]*)'

        try:
            with open(self.file_path, 'r') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:
                        continue

                    match = re.match(can_pattern, line)
                    if match:
                        timestamp = float(match.group(1))
                        interface = match.group(2)
                        can_id = int(match.group(3), 16)
                        data = match.group(4)

                        # 데이터를 바이트 배열로 변환
                        data_bytes = []
                        if data:
                            for i in range(0, len(data), 2):
                                if i + 1 < len(data):
                                    data_bytes.append(int(data[i:i + 2], 16))

                        parsed_entry = {
                            'timestamp': timestamp,
                            'interface': interface,
                            'can_id': can_id,
                            'can_id_hex': f"0x{can_id:03X}",
                            'data_hex': data,
                            'data_bytes': data_bytes,
                            'data_length': len(data_bytes),
                            'line_number': line_num
                        }

                        self.parsed_data.append(parsed_entry)
                    else:
                        print(f"Warning: Could not parse line {line_num}: {line}")

            # DataFrame 생성
            self.df = pd.DataFrame(self.parsed_data)
            self.df['category'] = self.df['can_id'].apply(self._categorize_can_id)
            self.df['time_diff'] = self.df['timestamp'].diff()

            print(f"총 {len(self.parsed_data)}개의 CAN 메시지를 로드했습니다.")

        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {self.file_path}")
        except Exception as e:
            print(f"파일 로딩 중 오류 발생: {e}")

    def load_dbc_file(self, dbc_file_path):
        """
        DBC 파일을 로드하여 정확한 CAN ID 분류 적용

        Args:
            dbc_file_path (str): DBC 파일 경로
        """
        try:
            print(f"DBC 파일 로딩 시도: {dbc_file_path}")
            # DBC 파일 파싱 로직 (cantools 라이브러리 사용 권장)
            # pip install cantools 필요
            import cantools

            db = cantools.database.load_file(dbc_file_path)

            # DBC에서 메시지 정보 추출
            dbc_categories = {}
            for message in db.messages:
                can_id = message.frame_id
                message_name = message.name

                # 메시지 이름 기반으로 카테고리 추정
                category = self._categorize_by_message_name(message_name)

                if category not in dbc_categories:
                    dbc_categories[category] = []
                dbc_categories[category].append(can_id)

            # 기존 추정값을 DBC 정보로 교체
            self.bmw_can_categories = dbc_categories
            print(f"DBC 파일에서 {len(db.messages)}개의 메시지 정보를 로드했습니다.")

        except ImportError:
            print("cantools 라이브러리가 필요합니다: pip install cantools")
        except Exception as e:
            print(f"DBC 파일 로딩 실패: {e}")
            print("추정값을 계속 사용합니다.")

    def _categorize_by_message_name(self, message_name):
        """메시지 이름 기반 카테고리 분류"""
        name_lower = message_name.lower()

        if any(keyword in name_lower for keyword in ['engine', 'motor', 'rpm', 'throttle']):
            return 'Engine'
        elif any(keyword in name_lower for keyword in ['gear', 'trans', 'shift']):
            return 'Transmission'
        elif any(keyword in name_lower for keyword in ['brake', 'abs', 'esp', 'airbag']):
            return 'Safety_Systems'
        elif any(keyword in name_lower for keyword in ['light', 'lamp', 'led']):
            return 'Lighting'
        elif any(keyword in name_lower for keyword in ['climate', 'hvac', 'temp']):
            return 'Climate'
        elif any(keyword in name_lower for keyword in ['radio', 'navi', 'media']):
            return 'Infotainment'
        elif any(keyword in name_lower for keyword in ['door', 'window', 'seat', 'mirror']):
            return 'Comfort'
        elif any(keyword in name_lower for keyword in ['body', 'bcm', 'power']):
            return 'Body_Electronics'
        else:
            return 'Unknown'

    def discover_can_patterns(self):
        """
        CAN ID 패턴을 자동으로 발견하여 분류 개선
        실제 데이터 패턴을 기반으로 CAN ID를 그룹핑
        """
        if self.df is None:
            return

        print("\n=== CAN ID 패턴 자동 발견 ===")

        # 메시지 빈도 기반 분류
        id_stats = self.df.groupby('can_id').agg({
            'timestamp': ['count', 'nunique'],
            'data_length': ['mean', 'std'],
            'time_diff': 'mean'
        }).round(4)

        id_stats.columns = ['msg_count', 'unique_times', 'avg_data_len', 'std_data_len', 'avg_interval']

        # 패턴 기반 분류
        discovered_patterns = {
            'High_Frequency': [],  # > 10Hz
            'Medium_Frequency': [],  # 1-10Hz
            'Low_Frequency': [],  # < 1Hz
            'Fixed_Length_8': [],  # 항상 8바이트
            'Variable_Length': [],  # 가변 길이
            'Periodic': [],  # 주기적
            'Event_Based': []  # 이벤트 기반
        }

        for can_id, stats in id_stats.iterrows():
            frequency = stats['msg_count'] / (self.df['timestamp'].max() - self.df['timestamp'].min())

            # 빈도 기반 분류
            if frequency > 10:
                discovered_patterns['High_Frequency'].append(can_id)
            elif frequency > 1:
                discovered_patterns['Medium_Frequency'].append(can_id)
            else:
                discovered_patterns['Low_Frequency'].append(can_id)

            # 데이터 길이 기반 분류
            if abs(stats['avg_data_len'] - 8) < 0.1 and stats['std_data_len'] < 0.1:
                discovered_patterns['Fixed_Length_8'].append(can_id)
            elif stats['std_data_len'] > 1:
                discovered_patterns['Variable_Length'].append(can_id)

            # 주기성 기반 분류
            if stats['avg_interval'] > 0 and not np.isnan(stats['avg_interval']):
                if stats['avg_interval'] < 0.2:  # 200ms 이하 주기
                    discovered_patterns['Periodic'].append(can_id)
                else:
                    discovered_patterns['Event_Based'].append(can_id)

        print("발견된 패턴:")
        for pattern, ids in discovered_patterns.items():
            if ids:
                print(f"{pattern}: {len(ids)}개 ID")
                print(f"  예시: {[f'0x{id:03X}' for id in ids[:5]]}")

        return discovered_patterns

    def _categorize_can_id(self, can_id):
        """CAN ID를 카테고리별로 분류"""
        for category, ids in self.bmw_can_categories.items():
            if can_id in ids:
                return category
        return 'Unknown'

    def get_basic_statistics(self):
        """기본 통계 정보 출력"""
        if self.df is None:
            print("데이터가 로드되지 않았습니다.")
            return

        print("=== 기본 통계 정보 ===")
        print(f"총 메시지 수: {len(self.df)}")
        print(f"고유 CAN ID 수: {self.df['can_id'].nunique()}")
        print(f"분석 시간 범위: {self.df['timestamp'].max() - self.df['timestamp'].min():.3f}초")
        print(f"평균 메시지 간격: {self.df['time_diff'].mean():.6f}초")
        print()

        # CAN ID별 메시지 수
        print("=== CAN ID별 메시지 빈도 ===")
        id_counts = self.df.groupby(['can_id_hex', 'category']).size().sort_values(ascending=False)
        print(id_counts.head(15))
        print()

        # 카테고리별 통계
        print("=== 카테고리별 통계 ===")
        category_stats = self.df.groupby('category').agg({
            'can_id': 'nunique',
            'timestamp': 'count',
            'data_length': 'mean'
        }).round(2)
        category_stats.columns = ['고유_ID수', '메시지수', '평균_데이터길이']
        print(category_stats)

    def analyze_message_patterns(self):
        """메시지 패턴 분석"""
        if self.df is None:
            return

        print("\n=== 메시지 패턴 분석 ===")

        # 주기적 메시지 탐지
        periodic_messages = {}
        for can_id in self.df['can_id'].unique():
            id_data = self.df[self.df['can_id'] == can_id]['timestamp'].values
            if len(id_data) > 5:
                intervals = np.diff(id_data)
                avg_interval = np.mean(intervals)
                std_interval = np.std(intervals)

                # 주기적 메시지 판단 (표준편차가 평균의 20% 이하)
                if std_interval < avg_interval * 0.2 and avg_interval > 0.001:
                    periodic_messages[can_id] = {
                        'avg_interval': avg_interval,
                        'frequency': 1 / avg_interval,
                        'count': len(id_data)
                    }

        print("주기적 메시지 (빈도 > 1Hz):")
        for can_id, info in sorted(periodic_messages.items(),
                                   key=lambda x: x[1]['frequency'], reverse=True):
            if info['frequency'] > 1:
                print(f"  0x{can_id:03X}: {info['frequency']:.1f}Hz "
                      f"(주기: {info['avg_interval'] * 1000:.1f}ms, 개수: {info['count']})")

    def analyze_data_patterns(self, can_id=None):
        """특정 CAN ID의 데이터 패턴 분석"""
        if self.df is None:
            return

        if can_id is None:
            # 가장 빈번한 CAN ID 선택
            can_id = self.df['can_id'].value_counts().index[0]

        print(f"\n=== CAN ID 0x{can_id:03X} 데이터 패턴 분석 ===")

        id_data = self.df[self.df['can_id'] == can_id].copy()
        print(f"메시지 수: {len(id_data)}")

        if len(id_data) == 0:
            print("해당 CAN ID의 데이터가 없습니다.")
            return

        # 데이터 길이 분석
        print(f"데이터 길이: {id_data['data_length'].value_counts().to_dict()}")

        # 바이트별 값 변화 분석
        if len(id_data) > 1:
            print("\n바이트별 값 변화 분석:")
            max_len = id_data['data_length'].max()

            for byte_pos in range(min(8, max_len)):  # 최대 8바이트까지
                byte_values = []
                for _, row in id_data.iterrows():
                    if byte_pos < len(row['data_bytes']):
                        byte_values.append(row['data_bytes'][byte_pos])

                if byte_values:
                    unique_values = len(set(byte_values))
                    min_val, max_val = min(byte_values), max(byte_values)
                    print(f"  바이트 {byte_pos}: 범위({min_val:02X}-{max_val:02X}), "
                          f"고유값 {unique_values}개")

                    # 변화가 있는 바이트는 상세 분석
                    if unique_values > 1 and unique_values < 20:
                        value_counts = Counter(byte_values)
                        print(f"    값 분포: {dict(value_counts.most_common(5))}")

    def detect_anomalies(self):
        """이상 패턴 탐지"""
        if self.df is None:
            return

        print("\n=== 이상 패턴 탐지 ===")

        # 1. 비정상적으로 긴 메시지 간격
        long_gaps = self.df[self.df['time_diff'] > 0.1]['time_diff'].dropna()
        if len(long_gaps) > 0:
            print(f"긴 메시지 간격 (>100ms): {len(long_gaps)}개")
            print(f"  최대 간격: {long_gaps.max():.3f}초")

        # 2. 특이한 데이터 길이
        length_counts = self.df['data_length'].value_counts()
        unusual_lengths = length_counts[length_counts < 10]
        if len(unusual_lengths) > 0:
            print(f"드문 데이터 길이: {unusual_lengths.to_dict()}")

        # 3. 특정 패턴 (모든 FF 또는 모든 00)
        all_ff_count = 0
        all_00_count = 0

        for _, row in self.df.iterrows():
            if row['data_hex']:
                if all(c == 'F' for c in row['data_hex']):
                    all_ff_count += 1
                elif all(c == '0' for c in row['data_hex']):
                    all_00_count += 1

        if all_ff_count > 0:
            print(f"모든 비트가 1인 메시지: {all_ff_count}개")
        if all_00_count > 0:
            print(f"모든 비트가 0인 메시지: {all_00_count}개")

    def generate_visualizations(self):
        """데이터 시각화 생성"""
        if self.df is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. CAN ID별 메시지 빈도
        top_ids = self.df['can_id_hex'].value_counts().head(10)
        axes[0, 0].bar(range(len(top_ids)), top_ids.values)
        axes[0, 0].set_title('상위 10개 CAN ID 메시지 빈도')
        axes[0, 0].set_xticks(range(len(top_ids)))
        axes[0, 0].set_xticklabels(top_ids.index, rotation=45)

        # 2. 시간에 따른 메시지 분포
        time_bins = np.linspace(self.df['timestamp'].min(),
                                self.df['timestamp'].max(), 50)
        axes[0, 1].hist(self.df['timestamp'], bins=time_bins, alpha=0.7)
        axes[0, 1].set_title('시간에 따른 메시지 분포')
        axes[0, 1].set_xlabel('시간 (초)')

        # 3. 카테고리별 메시지 수
        category_counts = self.df['category'].value_counts()
        axes[1, 0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('카테고리별 메시지 분포')

        # 4. 데이터 길이 분포
        length_counts = self.df['data_length'].value_counts().sort_index()
        axes[1, 1].bar(length_counts.index, length_counts.values)
        axes[1, 1].set_title('데이터 길이 분포')
        axes[1, 1].set_xlabel('데이터 길이 (바이트)')

        plt.tight_layout()
        plt.show()

    def export_analysis_report(self, output_file='can_analysis_report.txt'):
        """분석 결과를 파일로 출력"""
        if self.df is None:
            return

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("BMW CAN 데이터 분석 보고서\n")
            f.write("=" * 50 + "\n\n")

            # 기본 통계
            f.write(f"분석 파일: {self.file_path}\n")
            f.write(f"분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("기본 통계:\n")
            f.write(f"- 총 메시지 수: {len(self.df)}\n")
            f.write(f"- 고유 CAN ID 수: {self.df['can_id'].nunique()}\n")
            f.write(f"- 분석 시간 범위: {self.df['timestamp'].max() - self.df['timestamp'].min():.3f}초\n\n")

            # CAN ID별 상세 정보
            f.write("CAN ID별 상세 정보:\n")
            for can_id in sorted(self.df['can_id'].unique()):
                id_data = self.df[self.df['can_id'] == can_id]
                category = id_data['category'].iloc[0]
                f.write(f"- 0x{can_id:03X} ({category}): {len(id_data)}개 메시지\n")

        print(f"분석 보고서가 {output_file}에 저장되었습니다.")

    def interactive_analysis(self):
        """대화형 분석 메뉴"""
        while True:
            print("\n" + "=" * 50)
            print("BMW CAN 데이터 분석기")
            print("=" * 50)
            print("1. 기본 통계 정보")
            print("2. 메시지 패턴 분석")
            print("3. 특정 CAN ID 데이터 분석")
            print("4. 이상 패턴 탐지")
            print("5. 시각화 생성")
            print("6. 분석 보고서 출력")
            print("8. CAN ID 패턴 자동 발견")
            print("9. DBC 파일 로드")
            print("10. 종료")

            choice = input("\n선택하세요 (1-10): ").strip()

            if choice == '1':
                self.get_basic_statistics()
            elif choice == '2':
                self.analyze_message_patterns()
            elif choice == '3':
                can_id_hex = input("분석할 CAN ID (예: 0x175): ").strip()
                try:
                    can_id = int(can_id_hex, 16)
                    self.analyze_data_patterns(can_id)
                except ValueError:
                    print("올바른 16진수 형식으로 입력하세요.")
            elif choice == '4':
                self.detect_anomalies()
            elif choice == '5':
                self.generate_visualizations()
            elif choice == '7':
                filename = input("출력 파일명 (기본: can_analysis_report.txt): ").strip()
                if not filename:
                    filename = 'can_analysis_report.txt'
                self.export_analysis_report(filename)
            elif choice == '8':
                self.discover_can_patterns()
            elif choice == '9':
                dbc_file = input("DBC 파일 경로: ").strip()
                self.load_dbc_file(dbc_file)
                print("분석을 다시 실행합니다...")
                self.df['category'] = self.df['can_id'].apply(self._categorize_can_id)
            elif choice == '10':
                print("프로그램을 종료합니다.")
                break
            else:
                print("올바른 번호를 선택하세요.")


# 사용 예제
if __name__ == "__main__":
    # CAN 분석기 초기화 및 실행
    analyzer = BMWCANAnalyzer("260514_bmw_part_0001.candump")
    analyzer.load_and_parse_data()

    # 자동 분석 실행
    analyzer.get_basic_statistics()
    analyzer.analyze_message_patterns()
    analyzer.analyze_data_patterns()
    analyzer.detect_anomalies()

    # 대화형 분석 모드 (선택사항)
    # analyzer.interactive_analysis()