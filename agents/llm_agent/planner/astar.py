import heapq

def astar(start, goal, world):
    def h(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = [(0, start)]
    came_from = {}
    g = {start: 0}

    while open_set:
        _, cur = heapq.heappop(open_set)

        if cur == goal:
            return reconstruct_path(came_from, cur)

        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = cur[0] + dx, cur[1] + dy
            if not (0 <= nx < world.config.grid_width and 0 <= ny < world.config.grid_height):
                continue

            nxt = (nx, ny)
            tg = g[cur] + 1
            if tg < g.get(nxt, float("inf")):
                came_from[nxt] = cur
                g[nxt] = tg
                f = tg + h(nxt, goal)
                heapq.heappush(open_set, (f, nxt))

    return []


def reconstruct_path(came_from, cur):
    path = []
    while cur in came_from:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path