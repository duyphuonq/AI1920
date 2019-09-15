from util import PriorityQueue

frontier = PriorityQueue()
a = (2,3)
b = (2,4)
c = (2,5)
frontier.push((a, []), -2)
frontier.push((b, ['hi', 'hi']), 1)
frontier.push((c, ['ha','ha','ha']), 0)

pri, (state, actions) = frontier.pop()
print len(actions)