# Engine calibration runner

빠른 시뮬레이션을 위해 replay emission을 기본으로 끈 상태에서(n_games 만큼) 임의의 더미 팀/로스터/전술/역할을 생성해
`sim_game.simulate_game()`을 반복 실행하고, 팀-게임 샘플(=2*n_games)의 평균값을 JSON으로 저장합니다.

## 실행 (패키지 모드)
프로젝트 루트에서:

```bash
python -m engine.calibration.run --n_games 500 --seed 42 --style modern --out calib.json
```

## 옵션
- `--replay` : 리플레이 이벤트까지 포함(느리고 결과 파일 커짐)
- `--store_per_game` : 게임별 결과/입력까지 저장(매우 큼. 디버깅용)
- `--style` : 전술/로스터 방향성 프리셋(modern/motion/post/pace)

## 출력
- `league_avg_team_game`: 팀-게임(=한 팀의 한 경기) 기준 평균 스탯/카운트
- `league_avg_derived`: 평균 스탯 기반 파생 지표(PTS/100, FG%, etc.)
