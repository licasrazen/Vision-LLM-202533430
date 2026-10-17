scores = [[45, 88, 92],
          [30, 75, 60],
          [95, 40, 50]]

# TODO: List Comprehension을 활용해 50점 이상인 값만 통과시켜 담으세요.
passed_scores = [score for row in scores for score in row if score >= 50]
print("합격 점수들:", passed_scores)



students = ["짱구", "철수", "맹구"]
scores = [85, 92, 78]

# TODO: enumerate와 zip을 조합하거나 활용하여 80점 이상 학생 정보 출력
# TODO: 예상 출력 "1번 짱구 학생 (점수: 85점)"

for idx, (name, score) in enumerate(zip(students, scores), start=1):
  if score >= 80:
      print(f"{idx}번 {name}학생 (점수: {score}점)")


import numpy as np

np_arr = np.array([1, 2, 3, 4, 5])

# TODO: NumPy 연산 함수를 사용하여 배열을 정규화

np_arr_normalized = (np_arr-np.min(np_arr)/(np.max(np_arr)))
print(np_arr_normalized)

# TODO: NumPy 연산 함수를 사용하여 배열을 0.0 ~ 1.0 범위의 float32 배열로 변환
# TODO: 최종 배열의 최솟값 = 0.0, 최종 배열의 최댓값 = 1.0

np_arr_final = np_arr_normalized.astype(np.float32)
print(np_arr_final)
print(np_arr_final.dtype)

