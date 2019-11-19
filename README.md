# AI1920
## P1 : ./P1/search/search.py and ./P1/search/searchAgents.py
---
DFS: dùng Stack.

BFS: dùng Queue.

UCS, A*: dùng priority Queue.

---

## P2: ./P2/multiagent/multiAgents.py

Q1: [Reflex Agent](http://ai.berkeley.edu/multiagent.html#Q1).

Cải tiến class ReflexAgent trong multiAgents.py.

Giải pháp: Cải tiến hàm đánh giá evaluationFunction để pacman ưu tiên tìm kiếm thức ăn gần nhất và tránh ma.

```python
foodPos = newFood.asList()
        foodCount = len(foodPos)
        minDis = 10**6
        if foodCount == 0:
          minDis = 0
        else:
          distance = min([manhattanDistance(foodPos[i], newPos) + foodCount*100 for i in range(foodCount)])
          if distance < minDis:
            minDis = distance
        score = -minDis
        for i in range(len(newGhostStates)):
          ghostPos = successorGameState.getGhostPosition(i+1)
          if manhattanDistance(newPos,ghostPos)<=1 :
            score -= 10**6
        return score 
```
Q2:[Minimax](http://ai.berkeley.edu/multiagent.html#Q2)

Xây dựng thuật toán Minimax.
```python
numAgents = gameState.getNumAgents()
        #print numAgents
        scores = []
        def _minimax(s, iterCount):
          if iterCount >= self.depth*numAgents or s.isWin() or s.isLose():
            return self.evaluationFunction(s)
          if iterCount%numAgents != 0:    #is ghost
            result = 10**10
            for action in s.getLegalActions(iterCount%numAgents):
              successors = s.generateSuccessor(iterCount%numAgents, action)
              result = min(result, _minimax(successors, iterCount+1))
            return result
          else:   #is Pacman
            result = -10**10
            for action in s.getLegalActions(iterCount%numAgents):
              successors = s.generateSuccessor(iterCount%numAgents, action)
              result = max(result, _minimax(successors, iterCount+1))
              if iterCount == 0:
                scores.append(result)
            return result
        result = _minimax(gameState, 0)
        return gameState.getLegalActions(0)[scores.index(max(scores))]
```

Q3: [Alpha-Beta Prunning](http://ai.berkeley.edu/multiagent.html#Q3)

Xây dựng thuật toán Alpha-Beta
```python
numAgents = gameState.getNumAgents()
        scores = []
        def _alphaBeta(s, iterCount, alpha, beta):
          if iterCount >= self.depth*numAgents or s.isWin() or s.isLose():
            return self.evaluationFunction(s)
          if iterCount%numAgents != 0:    #Ghost min
            result = 10**10
            choose_min = iterCount%numAgents
            for action in s.getLegalActions(choose_min):
              successors = s.generateSuccessor(choose_min, action)
              #print "hihi"
              result = min(result, _alphaBeta(successors, iterCount+1, alpha, beta))
              beta = min(beta, result)
              if beta < alpha:
                break
            return result
          else:
            result = -10**10
            choose_max = iterCount%numAgents
            for action in s.getLegalActions(choose_max):
              successors = s.generateSuccessor(choose_max, action)
              #print "haha"
              result = max(result, _alphaBeta(successors, iterCount+1, alpha, beta))
              alpha = max(alpha, result)
              if iterCount == 0:
                scores.append(result)
              if beta < alpha:
                break
            return result
        result = _alphaBeta(gameState, 0, -10**20, 10**20)
        return gameState.getLegalActions(0)[scores.index(max(scores))]
```

Q4: [ExpectedMinimax](http://ai.berkeley.edu/multiagent.html#Q4)

Xây dựng thuật toán mới cho Agents đi những bước đi không hẳn là tối ưu.

```python
def _expectMinimax(s, iterCount):
          if iterCount >= self.depth*numAgents or s.isWin() or s.isLose(): #leaf node
            return self.evaluationFunction(s)
          if iterCount%numAgents != 0: #is Ghost 
            successorScore = []
            for a in s.getLegalActions(iterCount%numAgents):
              new_gameState = s.generateSuccessor(iterCount%numAgents,a)
              result = _expectMinimax(new_gameState, iterCount+1)
              successorScore.append(result)
            averageScore = sum([ float(x)/len(successorScore) for x in successorScore])
            return averageScore
          else: #is Pacman
            result = -10**10
            for a in s.getLegalActions(iterCount%numAgents):
              new_gameState = s.generateSuccessor(iterCount%numAgents,a)
              result = max(result, _expectMinimax(new_gameState, iterCount+1))
              if iterCount == 0:
                ActionScore.append(result)
            return result
      
        result = _expectMinimax(gameState, 0)
        return gameState.getLegalActions(0)[ActionScore.index(max(ActionScore))]
 ```
 Q5: [Evalution Function](http://ai.berkeley.edu/multiagent.html#Q5)
 
 Xây dựng các hàm để tính điểm tốt hơn, không chỉ dựa trên nước đi mà còn dựa trên vị trí của Pacman.
 
 Các hàm:
 
 ghostHunting - Đuổi ma khi ăn được viên nộ.
 ```python
 def _ghostHunting(gameState):
      score = 0
      for ghost in gameState.getGhostStates():
        disGhost = manhattanDistance(gameState.getPacmanPosition(), ghost.getPosition())
        if ghost.scaredTimer > 0:
          score += pow(max(8 - disGhost, 0), 2)
        else:
          score -= pow(max(7 - disGhost, 0), 2)
      return score 
```
foodGobbling - Tìm thức ăn gần nhất.
```python
def _foodGobbling(gameState):
      disFood = []
      for food in gameState.getFood().asList():
        disFood.append(1.0/manhattanDistance(gameState.getPacmanPosition(), food))
      if len(disFood)>0:
        return max(disFood)
      else:
        return 0
 ```
 pelletNabbing - Tìm viên nộ gần nhất
 ```python
 def _pelletNabbing(gameState):
      score = []
      for Cap in gameState.getCapsules():
        score.append(50.0/manhattanDistance(gameState.getPacmanPosition(), Cap))
      if len(score) > 0:
        return max(score)
      else:
        return 0
  ```
 
 
