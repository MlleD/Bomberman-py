import asyncio
import websockets
from websockets.exceptions import ConnectionClosedError
import json
from random import choice
import queue
from sys import argv
from math import inf
from copy import deepcopy

class Bomb:
    def __init__(self, x, y, idplayer, power):
        self.x = x
        self.y = y
        self.idplayer = idplayer
        self.power = power
        self.score = 0
    def set_score(self, score):
        self.score = score
    def __str__(self):
        s = "Bomb of p" + str(self.idplayer) + " at (" + str(self.x) + ", " + str(self.y)
        s += ")" + " with power " + str(self.power)
        return s

class Bombs:
    def __init__(self, bombs=None):
        self.bombs = set() if bombs is None else bombs
        self.current_score = nb_non_metal_cases
    def add(self, bomb: Bomb):
        self.bombs.add(bomb)
        bomb.set_score(self.current_score)
        self.current_score -= 1
    def copy(self):
        return Bombs(bombs=self.bombs.copy())
    def get(self, x, y):
        for bomb in self.bombs:
            if bomb.x == x and bomb.y == y:
                return bomb
    def pop(self, bomb):
        self.bombs.discard(bomb)
        if self.bombs == set():
            self.current_score = nb_non_metal_cases
    def __iter__(self):
        return iter(self.bombs)
    def __str__(self):
        s = 'Bombs:\n'
        for bomb in self.bombs:
            s += '- ' + str(bomb) + '\n'
        return s[:-1] # don't take the final \n

class Player:
    def __init__(self, name, id, xBlock, yBlock):
        self.name = name
        self.id = id
        self.xBlock = xBlock
        self.yBlock = yBlock
        self.max_nb_bombs = 1
        self.nb_bombs = 1
        self.bomb_power = 1
        self.dead = False
        self.stopped = True
    def decrement_nb_bombs(self):
        self.nb_bombs -= 1
    def die(self):
        self.dead = True
    def get_bomb_power(self):
        return self.bomb_power
    def get_coords(self):
        return self.xBlock, self.yBlock
    def get_id(self):
        return self.id
    def get_max_nb_bombs(self):
        return self.max_nb_bombs
    def get_name(self):
        return self.name
    def get_nb_bombs(self):
        return self.nb_bombs
    def increment_bomb_power(self):
        self.bomb_power += 1
    def increment_max_nb_bombs(self):
        self.max_nb_bombs += 1
    def increment_nb_bombs(self):
        self.nb_bombs += 1
    def change_stop_state(self):
        self.stopped = not self.stopped
    def update_coords(self, x, y):
        self.xBlock = x
        self.yBlock = y
    def __str__(self):
        s = "Player: name " + self.name + ", id " + str(self.id)
        s += ", xBlock " + str(self.xBlock) + ", yBlock " + str(self.yBlock)
        s += ", nb_bombs " + str(self.nb_bombs)
        return s

class Players:
    def __init__(self):
        self.dict = dict()
    def add(self, player: dict) -> None :
        self.dict[player["new_player"]] = Player(
            player["new_player"], int(player["id"]), int(player["xBlock"]), int(player["yBlock"])
        )
    def get(self, name: str):
        return self.dict.get(name)
    def get_from_id(self, id: int):
        for v in self.dict.values():
            if v.get_id() == id:
                return self.dict[v.get_name()]
        return None
    def __iter__(self):
        return iter(self.dict)
    def __len__(self):
        return len(self.dict)
    def __str__(self):
        s = "Players:\n"
        for k in self.dict:
            s += k + ": " + str(self.dict[k]) + '\n'
        return s
    
class Board:
    def __init__(self, board=None, m=[]):
        if board is None:
            self.board = []
            for r, row in enumerate(m):
                self.board.append([])
                for c, col in enumerate(row):
                    elt = '_' if col is None else col
                    self.board[r].append({elt})
        else:
            self.board = board
    def copy(self):
        return Board(board=deepcopy(self.board))
    def discoverable_neighbors(self, current_x, current_y):
        neighbors = all_neighbors(current_x, current_y)
        discoverable = [((nx, ny), self.is_brick(nx, ny)) for nx, ny in neighbors]
        return [c for c, t in discoverable if t == True]

    def exposed_cases_by_bomb(self, bomb):
        # Delta of the neighbor of (x, y) at the top, right, bottom, left
        delta = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        i_delta = 0
        cases = [(bomb.x, bomb.y)]

        for _ in range(len(delta)):
            current_x, current_y = bomb.x + delta[i_delta][0], bomb.y + delta[i_delta][1]
            count_one_brick = False
            while (self.within_borders(current_x, current_y)
            and manhattan_distance(current_x, current_y, bomb.x, bomb.y) <= bomb.power
            and not self.is_metal(current_x, current_y) and not count_one_brick):
                cases.append((current_x, current_y))
                if self.is_brick(current_x, current_y):
                    count_one_brick = True
                current_x += delta[i_delta][0]
                current_y += delta[i_delta][1]
            i_delta += 1

        return cases        

    def first_possible_move(self, moves, current_x, current_y, me, bombs, farming_mode=False):
        if moves.empty():
            return None, moves
        move_j = moves.get() # move in json
        move_p = json.loads(move_j) # move in python structure
        print('[first_possible_move] move_p', move_p)
        if (move_p.get('bombe') is not None) and ('bomb' in self.get(current_x, current_y)):
            return self.first_possible_move(moves, current_x, current_y, me, bombs)
        elif move_p.get('path') is not None:
            first = move_p['path'][0]
            last = move_p['path'][-1]
            if manhattan_distance(current_x, current_y, first[0], first[1]) > 1 and len(last) == 2:
                # find case of path with minimum manhattan distance to current case
                minimum_distance = inf
                minimum_index = 0
                for i, l in enumerate(move_p['path']):
                    distance = manhattan_distance(current_x, current_y, l[0], l[1])
                    if distance < minimum_distance:
                        minimum_distance = distance
                        minimum_index = i
                path = self.path(current_x, current_y, move_p['path'][minimum_index][0], move_p['path'][minimum_index][1])
                # concat the path from current_case to minimum distance case and the path from this case to last case
                modified_path = path[:-1] + move_p['path'][minimum_index:]
                print('[first_possible_move] modified_path', modified_path)
                # modify queue moves and put the move_path with modified path
                moves.queue.appendleft(move_path(move_p.get('id'), modified_path))
                # call recursively first_possible_move with (modified moves, current_x, current_y) as parameters
                return self.first_possible_move(moves, current_x, current_y, me, bombs)
            elif (len(last) == 3):
                index_slice = -1
                bomb_in = False
                # The bot will stop before a bomb case if a bomb is already on it
                # So we need to find the first bomb case in the path
                for [x, y] in move_p['path'][:-1]:
                    index_slice += 1
                    if 'bomb' in self.get(x, y):
                        bomb_in = True
                        break
                if not bomb_in and 'bomb' not in self.get(last[0], last[1]):
                    return move_j, moves
                # In farming_mode, only the bomb case is relevant so do not consider
                # the path if there is already a bomb on it
                if farming_mode:
                    return self.first_possible_move(moves, current_x, current_y, me, bombs, farming_mode)
                # Find the last case not exploded by bomb
                if bomb_in:
                    x, y = move_p['path'][index_slice]
                else:
                    x, y = last[0], last[1]
                exploded = self.exposed_cases_by_bomb(bombs.get(x, y))
                index_slice2 = -1
                for [x, y] in move_p['path'][:index_slice + 1]:
                    index_slice2 += 1
                    if (x, y) in exploded:
                        break
                if index_slice2 == 0:
                    print('[first_possible_move] bomb at first case of path')
                    return self.first_possible_move(moves, current_x, current_y, me, bombs)
                bomb = Bomb(current_x, current_x, move_p['id'], me.bomb_power)
                if self.path_to_shelter_if_add_bomb(current_x, current_y, bomb, bombs) == []:
                    print('[first_possible_move] no shelter if add bomb')
                    return self.first_possible_move(moves, current_x, current_y, me, bombs)
                # Put a bomb on the last case not exploded
                move_p['path'][index_slice2 - 1].append('bombe')
                print('[first_possible_move] modified bomb path', move_p['path'][:index_slice2])
                return move_path(move_p['id'], move_p['path'][:index_slice2]), moves
            elif len(last) == 2:
                contains_bonus = False
                bonus_names = ['bnbombeUP', 'bnspeedUP', 'bnflammeUP']
                for bonus_name in bonus_names:
                    if bonus_name in self.get(last[0], last[1]):
                        contains_bonus = True
                        break
                if not contains_bonus:
                    print('[first_possible_move] no bonus on the last case')
                    return self.first_possible_move(moves, current_x, current_y, me, bombs)
                elif not self.is_safe_path(move_p['path'], bombs):
                    print('[first_possible_move] no safe path to the next bonus')
                    moves.queue.appendleft(move_j)
                    return None, moves
        return move_j, moves

    def get(self, x, y):
        return self.board[y][x] if self.within_borders(x, y) else set()
    def is_bomb(self, x, y):
        return self.within_borders(x, y) and 'bomb' in self.board[y][x]
    def is_brick(self, x, y):
        return self.within_borders(x, y) and 'b' in self.board[y][x]
    def is_metal(self, x, y):
        return self.within_borders(x, y) and 'x' in self.board[y][x]
    def is_safe_case(self, case_x, case_y, forbidden_cases):
        return (case_x, case_y) not in forbidden_cases
    def is_safe_path(self, path, bombs):
        forbidden_cases = compute_forbidden_cases(self, bombs)
        return self.is_safe_path_with_forbidden_cases(path, forbidden_cases)
    def is_safe_path_with_forbidden_cases(self, path, forbidden_cases):
        safe = True
        for [x, y] in path:
            if (x, y) in forbidden_cases:
                safe = False
                break
        return safe
    def least_dangerous_path_to_shelter_multibomb(self, current_x, current_y, bombs: Bombs):
        forbidden_cases = compute_scored_forbidden_cases(self, bombs)
        return self.least_dangerous_path_to_shelter_from_forbidden_cases(current_x, current_y, forbidden_cases)
    def least_dangerous_path_to_shelter_from_forbidden_cases(self, current_x, current_y, forbidden_cases):
        print('[least_dangerous_path_to_shelter_from_forbidden_cases] forbidden cases', forbidden_cases) 
        if self.is_safe_case(current_x, current_y, forbidden_cases):
            return []
        cx, cy = current_x, current_y
        q = queue.Queue()
        visited = dict()
        ld = [] # least dangerous (x, y)
        min_d = inf # minimum dangerosity
        for nx, ny in self.reachable_neighbors(cx, cy):
            if not self.is_bomb(nx, ny):
                if forbidden_cases.get((nx, ny), 0) < min_d:
                    ld = [(nx, ny)]
                    min_d = forbidden_cases.get((nx, ny), 0)
                elif forbidden_cases.get((nx, ny), 0) == min_d:
                    ld.append((nx, ny))
        for nx, ny in ld:
            q.put((nx, ny, cx, cy)) # add neighbor with parent case (cx, cy)
            visited[(nx, ny)] = (cx, cy)
        print('[least_dangerous_path_to_shelter_from_forbidden_cases] queue', str(list(q.queue)))
        found = False        
        ax, ay, px, py = [-1] * 4 # (x,y) de l'abri et (x, y) du parent d'une case
        while not found and not q.empty():
            cx, cy, px, py = q.get()
            if (cx, cy) not in forbidden_cases:
                found = True 
            else:
                n = self.reachable_neighbors(cx, cy)
                n.remove((px, py))
                ld = []
                min_d = inf                
                for nx, ny in n:
                    if self.is_bomb(nx, ny):
                        continue
                    if visited.get((nx, ny)) is None or forbidden_cases.get(visited[(nx, ny)], 0) > forbidden_cases.get((cx, cy), 0):
                        visited[(nx, ny)] = (cx, cy)
                        q.put((nx, ny, cx, cy))
            print('[least_dangerous_path_to_shelter_from_forbidden_cases] queue', str(list(q.queue)))
        print('[least_dangerous_path_to_shelter_from_forbidden_cases] visited', str(visited))
        mypath = []
        while (cx, cy) != (current_x, current_y):
            mypath.append([cx, cy])
            cx, cy = visited[(cx, cy)]
        print('[least_dangerous_path_to_shelter_from_forbidden_cases] path', mypath[::-1])
        return mypath[::-1]
    def path(self, source_x, source_y, dest_x, dest_y):
        ns = self.reachable_neighbors(source_x, source_y)
        q = queue.Queue()
        for (nx, ny) in ns:
            q.put((nx, ny, source_x, source_y))
        found = False
        visited = dict()
        while not found and not q.empty():
            current_x, current_y, parent_x, parent_y = q.get()
            visited[(current_x, current_y)] = (parent_x, parent_y)
            if (current_x, current_y) != (dest_x, dest_y):
                ns = self.reachable_neighbors(current_x, current_y)
                ns.remove((parent_x, parent_y))
                for nx, ny in ns:
                    if (nx, ny) not in visited:
                        q.put((nx, ny, current_x, current_y))
            else:
                found = True
        path = []
        if not found:
            return path
        current_x, current_y = dest_x, dest_y
        while (current_x, current_y) != (source_x, source_y):
            path.append([current_x, current_y])
            current_x, current_y = visited[(current_x, current_y)]
        return path[::-1]

    def paths_to_bonuses(self, bonuses: list, source_x, source_y):
        paths = []
        current_x, current_y = source_x, source_y
        for bonus in bonuses:
            p = self.path(current_x, current_y, bonus['x'], bonus['y'])
            # If it's my bomb, there's always a path to the bonus so I directly add the path
            if p != []:
                paths.append(p)
            current_x, current_y = bonus['x'], bonus['y']
        return paths

    def path_to_reachable_case_with_max_discoverable_neighbors(self, source_x, source_y):
        theoritical_maximum = 3 # theoritical maximum discoverable neighbors
        ns = self.discoverable_neighbors(source_x, source_y)
        current_maximum = len(ns) # current maximum discoverable neighbors
        cases = [(source_x, source_y)] # cases with maximum_discoverable_neighbors
        q = queue.Queue()
        rs = self.reachable_neighbors(source_x, source_y)
        for (nx, ny) in rs:
            q.put((nx, ny, source_x, source_y))
        found = False
        visited = dict()
        while not found and not q.empty():
            current_x, current_y, parent_x, parent_y = q.get()
            visited[(current_x, current_y)] = (parent_x, parent_y)
            nb_ns = len(self.discoverable_neighbors(current_x, current_y))
            if nb_ns == theoritical_maximum:
                cases.clear()
                cases.append((current_x, current_y))
                found = True
            else:
                if nb_ns > current_maximum:
                    cases.clear()
                    current_maximum = nb_ns
                if nb_ns >= current_maximum:
                    cases.append(((current_x, current_y)))
                rs = self.reachable_neighbors(current_x, current_y)
                rs.remove((parent_x, parent_y))
                for nx, ny in rs:
                    can_add = True
                    for x,y,cx,cy in q.queue:
                        if (nx,ny) == (x,y):
                            can_add = False
                            break
                    if can_add:
                        q.put((nx, ny, current_x, current_y))
        path = []
        current_x, current_y = cases[0]
        while (current_x, current_y) != (source_x, source_y):
            path.append([current_x, current_y])
            current_x, current_y = visited[(current_x, current_y)]
        if len(path) == 0:
            path.append([source_x, source_y])
        return path[::-1]

    def path_to_reachable_case_with_max_discoverable_neighbors2(self, source_x, source_y, boundary):
        theoritical_maximum = 3 # theoritical maximum discoverable neighbors
        current_maximum = 0 # current maximum discoverable neighbors
        cmdn_x, cmdn_y = -1, -1 # case (x, y) with maximum_discoverable_neighbors 
        minimum_distance = max_manhattan_distance # minimum distance with the source (x, y)
        for case_x, case_y in boundary:
            nb_ds = len(self.discoverable_neighbors(case_x, case_y))
            if nb_ds > current_maximum:
                cmdn_x, cmdn_y = case_x, case_y
                minimum_distance = manhattan_distance(source_x, source_y, case_x, case_y)
                current_maximum = nb_ds
            elif nb_ds == current_maximum:
                distance = manhattan_distance(source_x, source_y, case_x, case_y)
                if distance < minimum_distance:
                    cmdn_x, cmdn_y = case_x, case_y
                    minimum_distance = distance
        path = self.path(source_x, source_y, cmdn_x, cmdn_y)
        if len(path) == 0:
            return [[source_x, source_y]]
        else:
            return path

    def path_to_shelter(self, current_x, current_y, bomb):
        # If I'm already safe
        safe = ((current_x != bomb.x) and (current_y != bomb.y))
        if safe or manhattan_distance(current_x, current_y, bomb.x, bomb.y) > bomb.power:
            return []
        cx, cy = current_x, current_y
        q = queue.Queue()
        visited = dict()
        for nx, ny in self.reachable_neighbors(cx, cy):
            q.put((nx, ny, 1, cx, cy)) # add neighbor with distance 1 and parent case(cx, cy)
            visited[(nx, ny)] = (cx, cy)
        found = False
        
        ax, ay, px, py = [-1] * 4 # (x,y) de l'abri et (x, y) du parent d'une case
        while not found and not q.empty():
            cx, cy, d, px, py = q.get()
            if ((cx != bomb.x) and (cy != bomb.y)) or (d > bomb.power):
                found = True 
            else:
                n = self.reachable_neighbors(cx, cy)
                n.remove((px, py))
                #put all reachables neighbors into the queue with dist_parent +1 and parent case (cx, cy)
                for nx, ny in n:
                    if (nx, ny) not in visited:
                        q.put((nx, ny, d + 1, cx, cy))
                        visited[(nx, ny)] = (cx, cy)
        print('[path_to_shelter] visited', str(visited))
        mypath = []
        while (cx, cy) != (current_x, current_y):
            mypath.append([cx, cy])
            cx, cy = visited[(cx, cy)]
        return mypath[::-1]
    def path_to_shelter_if_add_bomb(self, current_x, current_y, bomb, bombs):
        # Add the bomb on the copy of the board
        bcopy = self.copy()
        bcopy.update([[current_x, current_y, 'bomb']])
        # Copy bombs
        bombs_copy = bombs.copy()
        # Add bomb to the copy of bombs
        bombs_copy.add(bomb)
        # Return the path to shelter with multiple bombs
        return bcopy.path_to_shelter_multibomb(current_x, current_y, bombs_copy)
    def path_to_shelter_multibomb(self, current_x, current_y, bombs: Bombs):
        forbidden_cases = compute_forbidden_cases(self, bombs)
        return self.path_to_shelter_from_forbidden_cases(current_x, current_y, forbidden_cases)
    def path_to_shelter_from_forbidden_cases(self, current_x, current_y, forbidden_cases):
        print('[path_to_shelter_from_forbidden_cases] forbidden cases', forbidden_cases) 
        if self.is_safe_case(current_x, current_y, forbidden_cases):
            return []
        cx, cy = current_x, current_y
        q = queue.Queue()
        visited = dict()
        for nx, ny in self.reachable_neighbors(cx, cy):
            if not self.is_bomb(nx, ny):
                q.put((nx, ny, cx, cy)) # add neighbor with parent case (cx, cy)
                visited[(nx, ny)] = (cx, cy)
        print('[path_to_shelter_from_forbidden_cases] queue', str(list(q.queue)))
        found = False        
        ax, ay, px, py = [-1] * 4 # (x,y) de l'abri et (x, y) du parent d'une case
        while not found and not q.empty():
            cx, cy, px, py = q.get()
            if (cx, cy) not in forbidden_cases:
                found = True 
            else:
                n = self.reachable_neighbors(cx, cy)
                n.remove((px, py))
                # put all reachables neighbors into the queue with parent case (cx, cy)
                for nx, ny in n:
                    if visited.get((nx, ny)) is None and not self.is_bomb(nx, ny):
                        q.put((nx, ny, cx, cy))
                        visited[(nx, ny)] = (cx, cy)
            print('[path_to_shelter_from_forbidden_cases] queue', str(list(q.queue)))
        print('[path_to_shelter_from_forbidden_cases] visited', str(visited))
        mypath = []
        while (cx, cy) != (current_x, current_y):
            mypath.append([cx, cy])
            cx, cy = visited[(cx, cy)]
        print('[path_to_shelter_from_forbidden_cases] path', mypath[::-1])
        return mypath[::-1]

    def reachable_case(self, x, y):
        return self.within_borders(x, y) and 'b' not in self.board[y][x] \
            and 'x' not in self.board[y][x]
    def reachable_neighbors(self, current_x, current_y):
        neighbors = all_neighbors(current_x, current_y)
        reachable = [((nx, ny), self.reachable_case(nx, ny)) for nx, ny in neighbors]
        return [c for c, t in reachable if t == True]

    def safe_path(self, source_x, source_y, dest_x, dest_y, forbidden_cases):
        ns = self.reachable_neighbors(source_x, source_y)
        q = queue.Queue()
        for (nx, ny) in ns:
            if self.is_safe_case(nx, ny, forbidden_cases):
                q.put((nx, ny, source_x, source_y))
        found = False
        visited = dict()
        while not found and not q.empty():
            current_x, current_y, parent_x, parent_y = q.get()
            visited[(current_x, current_y)] = (parent_x, parent_y)
            if (current_x, current_y) != (dest_x, dest_y):
                ns = self.reachable_neighbors(current_x, current_y)
                ns.remove((parent_x, parent_y))
                for nx, ny in ns:
                    if (nx, ny) not in visited and self.is_safe_case(nx, ny, forbidden_cases):
                        q.put((nx, ny, current_x, current_y))
            else:
                found = True
        path = []
        if not found:
            return path
        current_x, current_y = dest_x, dest_y
        while (current_x, current_y) != (source_x, source_y):
            path.append([current_x, current_y])
            current_x, current_y = visited[(current_x, current_y)]
        return path[::-1]

    def update(self, cases: list):
        if cases == []:
            return
        for c in cases:
            if not self.within_borders(c[0], c[1]):
                continue
            # if len(c) == 2 <=> "empty case"
            if len(c) == 2:
                self.board[c[1]][c[0]].clear()
                self.board[c[1]][c[0]].add('_')
            else:
                if self.board[c[1]][c[0]] == {'_'}:
                    self.board[c[1]][c[0]].discard('_')
                    self.board[c[1]][c[0]].add(c[2])
                elif c[2].startswith('not'):
                    elt = c[2][3:]
                    self.board[c[1]][c[0]].discard(elt)
                    if len(self.board[c[1]][c[0]]) == 0:
                        self.board[c[1]][c[0]].add('_')
                elif c[2] not in self.board[c[1]][c[0]]:
                    self.board[c[1]][c[0]].add(c[2])
                    for elt in self.board[c[1]][c[0]].copy():
                        if elt.startswith('bn'):
                            self.board[c[1]][c[0]].discard(elt)
    def within_borders(self, x, y):
        return x >= 0 and y >= 0 and len(self.board) > y and len(self.board[y]) > x
    def __str__(self):
        # compute the str of board
        s = 'Board:\n'
        s += '[\n'
        for row in self.board:
            s += ' ' + str(row) + '\n'
        s += ']'
        return s

class AssocPlayerBoundary:
    def __init__(self, nbplayers):
        self.assoc = dict()
        for i in range(nbplayers):
            self.assoc[i] = i
    def change(self, idplayer, new_id_boundary):
        try:
            self.assoc[idplayer] = new_id_boundary
        except IndexError:
            print('[AssocPlayerBoundary.change] IndexError when accessing the player id')
            pass
    def get(self, idplayer):
        return self.assoc.get(idplayer)
    def items(self):
        return self.assoc.items()
    def pop(self, idplayer):
        self.assoc.pop(idplayer)
    def values(self):
        return self.assoc.values()
    def __iter__(self):
        return iter(self.assoc)
    def __str__(self):
        return str(self.assoc)
class Boundary:
    def __init__(self, cases: list):
        self.boundary = set(cases)
    def add(self, x, y):
        self.boundary.add((x, y))
    def copy(self):
        return deepcopy(self.boundary)
    def empty(self):
        return self.boundary == set()
    def path_to_nearest_not_dangerous_boundary_case (self, source_x, source_y, board, last_bomb, bombs):
        min_distance = inf
        smallest_path = []
        bombs_without_last = bombs.copy()
        if last_bomb is not None:
            # If I don't remove the last bomb, I can't escape
            bombs_without_last.pop(last_bomb)
        dangerous_cases = compute_forbidden_cases(board, bombs_without_last)
        print('[path_to_nearest_not_dangerous_boundary_case] dangerous_cases', dangerous_cases)
        for (x, y) in self.boundary:
            # if (x, y) == (source_x, source_y) or board.is_bomb(x, y):
            if board.is_bomb(x, y):
                continue
            path = board.safe_path(source_x, source_y, x, y, dangerous_cases)
            if path == []:
                continue
            distance = len(path)
            if (distance > 0) and (distance < min_distance):
                min_distance = distance
                smallest_path = path
        print('[path_to_nearest_not_dangerous_boundary_case] smallest_path', smallest_path)
        return smallest_path
    def pop(self, x, y):
        self.boundary.discard((x, y))
    def union(self, other):
        self.boundary = self.boundary.union(other)
    def __iter__(self):
        return iter(self.boundary)
    def __str__(self):
        return str(self.boundary)
class Boundaries:
    def __init__(self, players, board):
        self.boundaries = dict()
        nbplayers = len(players)
        self.assoc = AssocPlayerBoundary(nbplayers)
        for id_ in range(nbplayers):
            p = players.get_from_id(id_)
            self.boundaries[id_] = Boundary(board.reachable_neighbors(p.xBlock, p.yBlock))
    def get(self, id_boundary):
        return self.boundaries.get(id_boundary)
    def get_from_id_player(self, id_player):
        return self.boundaries.get(self.assoc.get(id_player))
    def get_all_id_enemy_with_same_boundary(self, id_player, players):
        ids = []
        id_boundary = self.assoc.get(id_player)
        for (idp, idb) in self.assoc.items():
            idp_player = players.get_from_id(idp)
            if idp_player.dead:
                continue
            idp_name = idp_player.get_name()
            if (idb == id_boundary) and (idp != id_player) and not is_allie(idp_name):
                ids.append(idp)
        return ids
    def is_in_same_boundary(self, id_player1, id_player2):
        return self.assoc.get(id_player1) == self.assoc.get(id_player2)
    def is_in_unique_boundary(self, id_player):
        id_boundary = self.assoc.get(id_player)
        all_ids = self.assoc.values()
        is_unique = True
        my_id = False
        for id_ in all_ids:
            if id_boundary == id_:
                if not my_id:
                    my_id = True
                else:
                    is_unique = False
                    break
        return is_unique
    def no_boundary(self):
        return len(self.boundaries) == 1 and self.boundaries[0].empty()
    def only_allies_in_boundary(self, id_player, allies_ids):
        id_boundary = self.assoc.get(id_player)
        for (idp, idb) in self.assoc.items():
            if (idb == id_boundary) and (idp not in allies_ids):
                return False
        return True
    def pop(self, id_boundary):
        self.boundaries.pop(id_boundary)
    def update(self, bomb: Bomb, tocases: list, board: Board):
        # Remove the bomb case from the boundary of the bomb's owner
        owner_id_boundary = self.assoc.get(bomb.idplayer)
        owner_boundary = self.boundaries[owner_id_boundary]
        owner_boundary.pop(bomb.x, bomb.y)
        for (case_x, case_y) in owner_boundary.copy():
            if len(board.discoverable_neighbors(case_x, case_y)) == 0:
                owner_boundary.pop(case_x, case_y)
        for c in tocases:
            # Add the brick cases exploded by the bomb
            border = len(board.discoverable_neighbors(c[0], c[1])) > 0
            if len(c) == 2 and border:
                owner_boundary.add(c[0], c[1])
        # Search if the owner's boundary can be merged with another boundary
        # The boundary with the highest id between the two will be deleted
        deleted_id_boundary = set()
        for id_ in self.boundaries:
            if id_ != owner_id_boundary:
                # Loop over the cases of the other's boundary to find the
                # nearest case and search a path from the bomb to this case
                nearest_case_x, nearest_case_y = -1, -1
                minimum_distance = max_manhattan_distance
                other_boundary = self.boundaries[id_]
                for case_x, case_y in other_boundary:
                    distance = manhattan_distance(bomb.x, bomb.y, case_x, case_y)
                    if distance < minimum_distance:
                        nearest_case_x, nearest_case_y = case_x, case_y
                        distance = minimum_distance
                path = board.path(bomb.x, bomb.y, nearest_case_x, nearest_case_y)
                if path == []:
                    continue
                # If there is a path, merge the highest id boundary into the other
                # then add the cases of the bridge between the two boundaries
                # to the receiver boundary then update the players id boundaries
                # with max_id to min_id
                else:
                    id_list = [id_, owner_id_boundary]
                    min_id = min(id_list)
                    receiver = self.get(min_id)
                    if receiver is None:
                        continue
                    max_id = max(id_list)
                    giver = self.get(max_id)
                    border = lambda x, y: board.discoverable_neighbors(case_x, case_y) != []
                    for (case_x, case_y) in giver.copy():
                        if not border(case_x, case_y):
                            giver.pop(case_x, case_y)
                    receiver.union(giver)
                    path_set = set()
                    for [case_x, case_y] in path:
                        if ((case_x, case_y) not in receiver) and border(case_x, case_y):
                            path_set.add((case_x, case_y))
                    receiver.union(path_set)
                    deleted_id_boundary.add(max_id)
                    for idplayer, idboundary in self.assoc.items():
                        if idboundary == max_id:
                            self.assoc.change(idplayer, min_id)
        for id_ in deleted_id_boundary:
            self.pop(id_)
    def __str__(self):
        s = 'Boundaries:\n'
        for idboundary, boundary in self.boundaries.items():
            s += '['+ str(idboundary) +'] : ' + str(boundary) + '\n'
        s += 'Assoc Player-Boundary:\n' + str(self.assoc) + '\n'
        return s

# Json-encoded actions to send
def move_dir(id, direction):
    return json.dumps({'id': id, 'move': direction})
def put_bomb(id):
    return json.dumps({'bombe': id})
def move_path(id, path):
    return json.dumps({'id': id, 'path': path})

# Fonctions support
def to_cases(data: dict):
    cases = []
    for d in data['dead']:
        cases.append([d['x'], d['y'], 'notp' + str(d['whoIS'])])
    for x in range(data['x'] - data['infX'], data['x'] + data['supX'] + 1):
        cases.append([x, data['y']])
    for y in range(data['y'] - data['infY'], data['y'] + data['supY'] + 1):
        if y == data['y']:
            continue
        cases.append([data['x'], y])
    for bonus in data['bonus']:
        cases.append([bonus['x'], bonus['y'], 'bn' + bonus['type']])
    return cases
def ended_game(msg: dict) -> bool:
    return (msg.get('winner') is not None) or (msg.get('equalGame') is not None)
def all_neighbors(current_x, current_y):
    return [
            (current_x, current_y - 1), (current_x + 1, current_y),
            (current_x, current_y + 1), (current_x - 1, current_y)
        ]
def manhattan_distance(source_x, source_y, dest_x, dest_y):
    return abs(source_x - dest_x) + abs(source_y - dest_y)
max_manhattan_distance = 34 # board with height 21 and width 15
def compute_forbidden_cases(board, bombs):
    forbidden_cases = set()
    for bomb in bombs:
        for exposed in board.exposed_cases_by_bomb(bomb):
            forbidden_cases.add(exposed)
    return forbidden_cases
def retrieve_player_id(message):
    player_id = None
    possible_id_field = ['alert_bombe', 'boom', 'id', 'takenBY']
    len_pif = len(possible_id_field)
    i = 0
    while (player_id is None) and (i < len_pif):
        player_id = message.get(possible_id_field[i])
        i += 1
    return player_id
def my_boom_common_processing(res, bonuses, nth_bomb, moves: queue.Queue, board: Board, me: Player):
    nth_bomb += 1
    print('[nth bomb]', nth_bomb)
    if res.get('bonus') != []:
        bonuses += res.get('bonus')
    less_than_max_bombs = me.get_nb_bombs() < me.get_max_nb_bombs()
    if not less_than_max_bombs:
        paths = board.paths_to_bonuses(bonuses, me.xBlock, me.yBlock)
        print('[paths to bonuses] bonuses', bonuses)
        print('[paths to bonuses] paths', paths)
        for path in paths:
            moves.put(move_path(me.get_id(), path))
    return bonuses, nth_bomb, moves, less_than_max_bombs
def path_to_nearest_not_dangerous_accessible_enemy(board, player_x, player_y, player_id, players, boundaries, forbidden_cases):
    all_ids = boundaries.get_all_id_enemy_with_same_boundary(player_id, players)
    if all_ids == []:
        return []
    smallest_path = []
    len_smallest_path = inf
    for idp in all_ids:
        o_x, o_y = players.get_from_id(idp).get_coords()
        path = board.path(player_x, player_y, o_x, o_y)
        len_path = len(path)
        if len_path < len_smallest_path:
            safe = True
            for [x, y] in path:
                if (x, y) in forbidden_cases:
                    safe = False
                    break
            if not safe:
                continue
            smallest_path = path
            len_smallest_path = len_path
    return smallest_path
nb_non_metal_cases = 232
def compute_scored_forbidden_cases(board, bombs):
    forbidden_cases = dict()
    for bomb in bombs:
        for exposed in board.exposed_cases_by_bomb(bomb):
            if forbidden_cases.get(exposed) is None or forbidden_cases.get(exposed) < bomb.score:
                forbidden_cases[exposed] = bomb.score
    return forbidden_cases
def is_allie(player_name):
    return 'doro' in player_name
async def play(websocket, myname, nbplayers_begin):
    players = Players()
    map_game = None
    allies_ids = set()
    while map_game is None:
        msg = json.loads(await websocket.recv())
        if type(msg) is int:
            continue
        elif msg.get("new_player") is not None:
            if players.get(msg.get('new_player')) is None:
                print("I'm in the game", msg)
                players.add(msg)
                if is_allie(msg.get('new_player')):
                    allies_ids.add(msg.get('id'))
        else:
            map_game = msg
    print(players)

    board = Board(m=map_game['map'])
    for p in players:
        infos = players.get(p)
        board.update([[infos.xBlock, infos.yBlock, 'p'+str(infos.id)]])
    print(board)

    myid = players.get(myname).id
    bombs = Bombs()

    # First move
    p = players.get(myname)
    possible = board.reachable_neighbors(p.xBlock, p.yBlock)
    i_bomb = choice([0, 1])
    x_bomb, y_bomb = possible[i_bomb]
    await websocket.send(move_path(myid, [[x_bomb, y_bomb, "bombe"]]))

    moves = queue.Queue()
    currently_stopped = False
    received_notif_my_alert_bomb = False
    my_last_bomb = None
    nth_bomb = 0
    bonuses = []
    shelter_path_from_last_bomb = []
    had_my_first_bonus = False
    my_first_bonus_is_bombup = False
    blocked_second_bomb_after_my_first_bonus_bombup = False

    boundaries = Boundaries(players, board)

    in_game = True

    while in_game:
        res = json.loads(await websocket.recv())
        print(res)
        print(board)
        if type(res) is int:
            print('[Server sent int] int =', res)
            continue
        elif ended_game(res):
            print(board)
            print('[nth bomb]', nth_bomb)
            in_game = False
        elif res.get('alert_bombe') is not None:
            bomb = Bomb(res.get('x'), res.get('y'), res.get('alert_bombe'), res.get('puissance'))
            bombs.add(bomb)
            board.update([[res.get('x'), res.get('y'), 'bomb']])
            print('[alert_bombe] added bomb: ', str(bombs.get(bomb.x, bomb.y)))
            owner_bomb = players.get_from_id(res.get('alert_bombe'))
            owner_bomb.decrement_nb_bombs()
            old_x, old_y = owner_bomb.get_coords()
            if (old_x, old_y) != (res.get('x'), res.get('y')):
                id_ = str(res.get('alert_bombe'))
                board.update([
                    [old_x, old_y, 'notp' + id_],
                    [res.get('x'), res.get('y'), 'p' + id_]
                ])
                owner_bomb.update_coords(res.get('x'), res.get('y'))
        elif res.get('boom') is not None:
            c = to_cases(res)
            print('[to_cases] c', c)
            bomb = bombs.get(res.get('x'), res.get('y'))
            if bomb is None:
                bomb = Bomb(res.get('x'), res.get('y'), res.get('boom'), res.get('puissance'))
            print('[After boom] bomb', bomb)
            print(boundaries)
            board.update(c)
            boundaries.update(bomb, c, board)
            bombs.pop(bomb)
            for dead in res.get('dead'):
                players.get_from_id(dead.get('whoIS')).die()
                print('[dead] player_id', dead.get('whoIS'))
            owner_bomb = players.get_from_id(res.get('boom'))
            if owner_bomb is not None:
                owner_bomb.increment_nb_bombs()
                print('[After boom] owner\'s bomb nb bombs', str(owner_bomb.get_nb_bombs()))
                print('[After boom] owner\'s bomb max bombs', str(owner_bomb.get_max_nb_bombs()))
        elif res.get('move') is not None:
            p = players.get_from_id(res.get('id'))
            old_x, old_y = p.get_coords()
            new_x, new_y = res.get('xBlock'), res.get('yBlock')
            if (old_x, old_y) != (new_x, new_y):
                id_ = str(res.get('id'))
                board.update([
                    [old_x, old_y, 'notp' + id_],
                    [new_x, new_y, 'p' + id_]
                ])
                players.get_from_id(res.get('id')).update_coords(new_x, new_y)
        elif res.get('takenBY') is not None:
            board.update([[res.get('x'), res.get('y'), 'notbn' + res.get('bnType')]])
            p = players.get_from_id(res.get('takenBY'))
            # Update the coords if different
            p_x, p_y = p.get_coords()
            if (p_x, p_y) != (res.get('x'), res.get('y')):
                id_ = str(res.get('takenBY'))
                board.update([
                    [p_x, p_y, 'notp' + id_],
                    [res.get('x'), res.get('y'), 'p' + id_]
                ])
                p.update_coords(res.get('x'), res.get('y'))
            if res.get('bnType') == 'bombeUP':
                p.increment_nb_bombs()
                p.increment_max_nb_bombs()
                print('[After taken bonus bomb] owner\'s bomb nb bombs', str(p.get_nb_bombs()))
                print('[After taken bonus bomb] owner\'s bomb max bombs', str(p.get_max_nb_bombs()))
            elif res.get('bnType') == 'flammeUP':
                p.increment_bomb_power()
            # Checks if it's my first bonus and if it is a bomb up:
            if not had_my_first_bonus and res.get('takenBY') == myid:
                had_my_first_bonus = True
                if res.get('bnType') == 'bombeUP':
                    my_first_bonus_is_bombup = True
                    print('[my first bonus is bombup]', my_first_bonus_is_bombup)

        player_id = retrieve_player_id(res)
        if player_id is None or players.get_from_id(myid) is None:
            continue

        if (player_id == myid):
            me = players.get_from_id(myid)
            if boundaries.is_in_unique_boundary(player_id) or boundaries.only_allies_in_boundary(myid, allies_ids):
                if res.get('alert_bombe') is not None:
                    received_notif_my_alert_bomb = True
                    my_last_bomb = bombs.get(res.get('x'), res.get('y'))
                    if currently_stopped:
                        me = players.get_from_id(myid)
                        current_x, current_y = me.get_coords()
                        if me.get_nb_bombs() == 0: # Je vais m'abriter
                            print('[received_notif_my_alert_bomb & currently_stopped] curr_x, curr_y',current_x, current_y)
                            path = board.least_dangerous_path_to_shelter_multibomb(current_x, current_y, bombs)
                            print('[shelter_path, l326]', path)
                            if path != []:
                                await websocket.send(move_path(myid, path))
                        else:
                            print('[Alert my bomb & currently stopped & not last bomb]')
                            my_boundary = boundaries.get_from_id_player(myid)
                            print('[my_boundary]')
                            print(my_boundary)
                            print('[current_x, current_y]', current_x, current_y)
                            path_to_next_bomb = my_boundary.path_to_nearest_not_dangerous_boundary_case(
                                current_x, current_y, board, my_last_bomb, bombs)
                            print('[path_to_next_bomb]', path_to_next_bomb)
                            next_bomb_x, next_bomb_y, my_next_bomb, shelter_path_from_next_bomb = [None] * 4
                            forbidden_cases = compute_scored_forbidden_cases(board, bombs)
                            if path_to_next_bomb != []:
                                [next_bomb_x, next_bomb_y] = path_to_next_bomb[-1]
                                my_next_bomb = Bomb(next_bomb_x, next_bomb_y, myid, me.get_bomb_power())
                                shelter_path_from_next_bomb = []
                                if shelter_path_from_last_bomb != [] \
                                and board.is_safe_path_with_forbidden_cases(shelter_path_from_last_bomb, forbidden_cases) \
                                and manhattan_distance( \
                                    current_x, current_y, \
                                    shelter_path_from_last_bomb[0][0], shelter_path_from_last_bomb[0][1]) == 1:
                                        shelter_path_from_next_bomb = shelter_path_from_last_bomb
                                else:
                                    shelter_path_from_next_bomb = board.path_to_shelter_if_add_bomb(
                                    next_bomb_x, next_bomb_y, my_next_bomb, bombs)
                                print('[Alert_bombe] compute next bomb: (x, y)', next_bomb_x, next_bomb_y)
                                print('[Alert_bombe] compute next bomb: shelter_path', shelter_path_from_next_bomb)
                                if my_first_bonus_is_bombup and not blocked_second_bomb_after_my_first_bonus_bombup:
                                    blocked_second_bomb_after_my_first_bonus_bombup = True
                                    # Must recompute the shelter path and cannot use the shelter path to next bomb
                                    # because the current case has a manhattan distance of 2 with the first
                                    # case of the shelter path to next bomb
                                    path = board.least_dangerous_path_to_shelter_from_forbidden_cases(current_x, current_y, forbidden_cases)
                                    print('[Block second bomb after first bonus bombup] least_dangerous_path_to_shelter_from_forbidden_cases', path)
                                    if path != []:
                                        await websocket.send(move_path(myid, path))
                                elif shelter_path_from_next_bomb != []:
                                    path_to_next_bomb[-1].append('bombe')
                                    print('[After alert_bombe] path_to_next_bomb (boundary case)',  path_to_next_bomb)
                                    await websocket.send(move_path(myid, path_to_next_bomb))
                                elif shelter_path_from_last_bomb != []:
                                    print('[After alert_bombe] shelter_path_from_last_bomb', shelter_path_from_last_bomb)
                                    await websocket.send(move_path(myid, shelter_path_from_last_bomb))
                                    shelter_path_from_last_bomb = []
                                else:
                                    path = board.least_dangerous_path_to_shelter_from_forbidden_cases(current_x, current_y, forbidden_cases)
                                    print('[After alert_bombe] least_dangerous_path_to_shelter_from_forbidden_cases', path)
                                    if path != []:
                                        await websocket.send(move_path(myid, path))
                            else:
                                path = board.least_dangerous_path_to_shelter_from_forbidden_cases(current_x, current_y, forbidden_cases)
                                print('[After alert_bombe & no path_to_next_bomb] least_dangerous_path_to_shelter_from_forbidden_cases', path)
                                if path != []:
                                    await websocket.send(move_path(myid, path))
                        currently_stopped = False
                elif res.get('boom') is not None:
                    me = players.get_from_id(myid)
                    if (me is None) or (len(players) == 1 and nbplayers_begin > 1):
                        continue
                    elif boundaries.no_boundary():
                        continue
                    elif res.get('boom') == myid:
                        bonuses, nth_bomb, moves, less_than_max_bombs = my_boom_common_processing(
                            res, bonuses, nth_bomb, moves, board, me
                        )
                        if less_than_max_bombs:
                            continue
                        x, y = me.xBlock, me.yBlock
                        if len(bonuses) > 0:
                            last_bonus = bonuses[-1]
                            x, y = last_bonus.get('x'), last_bonus.get('y')
                            bonuses.clear()
                        boundary = boundaries.get_from_id_player(myid)
                        path = board.path_to_reachable_case_with_max_discoverable_neighbors2(x, y, boundary)
                        print("[path to reachable case]", path)
                        if path[0] == [x, y]: # if reachable case is the last bonus case
                            last_bonus_path = moves.queue.pop()
                            last_bonus_path_py = json.loads(last_bonus_path)
                            path = last_bonus_path_py['path']
                        path[-1].append('bombe')
                        moves.put(move_path(myid, path))
                        print('[reaction to my boom] moves queue', list(moves.queue))
                        if not moves.empty():
                            possible_move, moves = board.first_possible_move(moves, me.xBlock, me.yBlock, me, bombs)
                            print('[reaction to my boom] possible_move', possible_move)
                            print('[reaction to my boom] next moves', list(moves.queue))
                            if possible_move is not None:
                                await websocket.send(possible_move)
                                continue
                        boundary = boundaries.get_from_id_player(myid)
                        path = boundary.path_to_nearest_not_dangerous_boundary_case(me.xBlock, me.yBlock, board, None, bombs)
                        print('[path_to_nearest_not_dangerous_boundary_case] path', path)
                        if path != []:
                            path[-1].append('bombe')
                            await websocket.send(move_path(myid, path))

                    else:
                        print('[Other\'s bomb exploded]')
                        if not moves.empty():
                            possible_move, moves = board.first_possible_move(moves, me.xBlock, me.yBlock, me, bombs)
                            print('[Moves not empty] possible_move', possible_move)
                            print('[Moves not empty] next moves', list(moves.queue))
                            if possible_move is not None:
                                await websocket.send(possible_move)
                                continue
                        print('[moves empty or impossible]')
                        x, y = me.xBlock, me.yBlock
                        boundary = boundaries.get_from_id_player(myid)
                        path = board.path_to_reachable_case_with_max_discoverable_neighbors2(x, y, boundary)
                        print("[path to reachable case]", path)
                        if path != []:
                            path[-1].append('bombe')
                            await websocket.send(move_path(myid, path))                        

                elif res.get('move') is not None:
                    if res.get('move') == 'stop' and res.get('id') == myid:
                        currently_stopped = True

                        p = players.get_from_id(myid)
                        current_x, current_y = p.get_coords()
                        if p.get_nb_bombs() == 0:
                            path = board.least_dangerous_path_to_shelter_multibomb(current_x, current_y, bombs)
                            print('[Stopped after alert for my last bomb] path to shelter', path)
                            if len(path) > 0: # if I'm not safe at my current position
                                await websocket.send(move_path(myid, path))
                            received_notif_my_alert_bomb = False
                        elif not moves.empty():
                            print('[Stopped] moves', list(moves.queue))
                            possible_move, moves = board.first_possible_move(moves, current_x, current_y, p, bombs)
                            print('[Stopped] possible_move', possible_move)
                            print('[Stopped] next moves', list(moves.queue))
                            if possible_move is None:
                                path = board.least_dangerous_path_to_shelter_multibomb(current_x, current_y, bombs)
                                print('[Stopped] path to shelter', path)
                                if path != []: # if I'm not safe at my current position
                                    await websocket.send(move_path(myid, path))
                            else:
                                print('[Stopped] possible_move sent')
                                await websocket.send(possible_move)
                        else:
                            # if I'm safe
                            print('[Stopped] check if my current case is safe')
                            forbidden_cases = compute_scored_forbidden_cases(board, bombs)
                            if not board.is_safe_case(current_x, current_y, forbidden_cases):
                                path = board.path_to_shelter_from_forbidden_cases(current_x, current_y, forbidden_cases)
                                print('[Stopped] not safe, path_to_shelter', path)
                                if path != []:
                                    await websocket.send(move_path(myid, path))
                                continue
                            print('[Stopped] my current case is safe')
                            my_boundary = boundaries.get_from_id_player(myid)
                            path_to_next_bomb = my_boundary.path_to_nearest_not_dangerous_boundary_case(
                                current_x, current_y, board, my_last_bomb, bombs)
                            print('[Stopped] path_to_next_bomb', path_to_next_bomb)
                            next_bomb_x, next_bomb_y, my_next_bomb, shelter_path_from_next_bomb = [None] * 4
                            if path_to_next_bomb != []:
                                [next_bomb_x, next_bomb_y] = path_to_next_bomb[-1]
                                my_next_bomb = Bomb(next_bomb_x, next_bomb_y, myid, p.get_bomb_power())
                                shelter_path_from_next_bomb = []
                                if shelter_path_from_last_bomb != []:
                                    first_case_x, first_case_y = shelter_path_from_last_bomb[0]
                                    if manhattan_distance(current_x, current_y, first_case_x, first_case_y) == 1 \
                                    and board.is_safe_path_with_forbidden_cases(shelter_path_from_last_bomb, forbidden_cases):
                                        shelter_path_from_next_bomb = shelter_path_from_last_bomb
                                else:
                                    shelter_path_from_next_bomb = board.path_to_shelter_if_add_bomb(
                                    next_bomb_x, next_bomb_y, my_next_bomb, bombs)
                                print('[Currently stopped & has path_to_next_bomb] compute next bomb: (x, y)', next_bomb_x, next_bomb_y)
                                print('[Currently stopped & has path_to_next_bomb] compute next bomb: shelter_path', shelter_path_from_next_bomb)
                            if not shelter_path_from_next_bomb:
                                # Recompute the path because the shelter_path_from_last_bomb may not be relevant
                                print('[No shelter from next bomb nor last bomb]')
                                print('[Stopped] current (x, y)', current_x, current_y)
                                path = board.least_dangerous_path_to_shelter_from_forbidden_cases(current_x, current_y, forbidden_cases)
                                print('[Stopped] path to shelter', path)
                                if path != []: # if I'm not safe at my current position
                                    await websocket.send(move_path(myid, path))
                                else:
                                    path = my_boundary.path_to_nearest_not_dangerous_boundary_case(current_x, current_y, board, None, bombs)
                                    print('[Stopped] path_to_nearest_not_dangerous_boundary_case', path)
                                    if path != []:
                                        path[-1].append('bombe')
                                        await websocket.send(move_path(myid, path))
                            elif p.get_nb_bombs() < p.get_max_nb_bombs():
                                print('[Stopped and has still at least one available bomb]')
                                shelter_path_from_last_bomb = shelter_path_from_next_bomb
                                if path_to_next_bomb != [] and not my_first_bonus_is_bombup:
                                    path_to_next_bomb[-1].append('bombe')
                                    # bombs.add(my_next_bomb)
                                    print('[Stopped] path_to_next_bomb', path_to_next_bomb)
                                    await websocket.send(move_path(myid, path_to_next_bomb))
                                else:
                                    print('[Stopped after my alert bomb] current (x, y)', current_x, current_y)
                                    path = board.least_dangerous_path_to_shelter_from_forbidden_cases(current_x, current_y, forbidden_cases)
                                    print('[Stopped after my alert bomb] path to shelter', path)
                                    if len(path) > 0: # if I'm not safe at my current position
                                        await websocket.send(move_path(myid, path))
            elif res.get('boom') is not None:
                if me is None:
                    continue
                bonuses, nth_bomb, moves, less_than_max_bombs = my_boom_common_processing(
                    res, bonuses, nth_bomb, moves, board, me
                )
                print('[my_boom_common_processing] bonuses', bonuses)
                print('[my_boom_common_processing] nth_bomb', nth_bomb)
                print('[my_boom_common_processing] moves', list(moves.queue))
                print('[my_boom_common_processing] less_than_max_bombs', less_than_max_bombs)
                if less_than_max_bombs:
                    continue
                x, y = me.xBlock, me.yBlock
                if len(bonuses) > 0:
                    last_bonus = bonuses[-1]
                    x, y = last_bonus.get('x'), last_bonus.get('y')
                    bonuses.clear()
                forbidden_cases = compute_forbidden_cases(board, bombs)
                path = path_to_nearest_not_dangerous_accessible_enemy(board, x, y, myid, players, boundaries, forbidden_cases)
                print('[path_to_nearest_not_dangerous_accessible_enemy] path', path)
                if path != []:
                    path[-1].append('bombe')
                    moves.put(move_path(myid, path))
                print('[reaction to my last boom] moves queue', list(moves.queue))
                if not moves.empty():
                    possible_move, moves = board.first_possible_move(moves, me.xBlock, me.yBlock, me, bombs)
                    print('[reaction to my last boom] possible_move', possible_move)
                    print('[reaction to my last boom] moves', list(moves.queue))
                    if possible_move is not None:
                        await websocket.send(possible_move)
            elif res.get('alert_bombe') is not None:
                current_x, current_y = players.get_from_id(myid).get_coords()
                path = board.least_dangerous_path_to_shelter_multibomb(current_x, current_y, bombs)
                print('[multiplayer, shelter_path_multibomb]', path)
                if path != []:
                    await websocket.send(move_path(myid, path))
            elif res.get('move') is not None and res.get('move') == 'stop':
                print('[Multiplayer mode & I\'m stopped] moves queue', list(moves.queue))
                print('[Multiplayer mode & I\'m stopped] my nb_bombs & max_nb_bombs', me.get_nb_bombs(), me.get_max_nb_bombs())
                forbidden_cases = compute_scored_forbidden_cases(board, bombs)
                is_safe = board.is_safe_case(res.get('xBlock'), res.get('yBlock'), forbidden_cases)
                print('[Multiplayer mode & I\'m stopped] current case is safe', is_safe)
                x, y = me.get_coords()
                if not is_safe:
                    path = board.least_dangerous_path_to_shelter_from_forbidden_cases(x, y, forbidden_cases)
                    print('[Multiplayer mode & I\'m stopped & not safe] path to shelter', path)
                    if path != []:
                        await websocket.send(move_path(myid, path))
                elif (me.get_nb_bombs() == me.get_max_nb_bombs()) and moves.empty():
                    path = path_to_nearest_not_dangerous_accessible_enemy(board, x, y, myid, players, boundaries, forbidden_cases)
                    print('[path_to_nearest_not_dangerous_accessible_enemy] path', path)
                    if path != []:
                        path[-1].append('bombe')
                        await websocket.send(move_path(myid, path))
                elif not moves.empty():
                    possible_move, moves = board.first_possible_move(moves, x, y, me, bombs)
                    print('[Multiplayer mode & I\'m stopped] possible_move', possible_move)
                    print('[Multiplayer mode & I\'m stopped] moves queue', list(moves.queue))
                    if possible_move is not None:
                        await websocket.send(possible_move)

        else :
            if not boundaries.is_in_same_boundary(myid, player_id):
                continue
            me = players.get_from_id(myid)
            x, y = me.get_coords()
            if res.get('alert_bombe') is not None:
                path = board.least_dangerous_path_to_shelter_multibomb(x, y, bombs)
                if path == []:
                    continue
                print('[Alert other\'s bomb] shelter path', path)
                await websocket.send(move_path(myid, path))
            elif res.get('move') == 'stop' and player_id not in allies_ids:
                forbidden_cases = compute_forbidden_cases(board, bombs)
                path = board.safe_path(x, y, res.get('xBlock'), res.get('yBlock'), forbidden_cases)
                print('[Enemy p' + str(res.get('id')) + ' stopped] path', path)
                if path == [] and (res.get('xBlock'), res.get('yBlock')) != (x, y):
                    continue
                bomb = Bomb(res.get('xBlock'), res.get('yBlock'), myid, me.get_bomb_power())
                shelter_path = board.path_to_shelter_if_add_bomb(res.get('xBlock'), res.get('yBlock'), bomb, bombs)
                if shelter_path == []:
                    continue
                if path != []:
                    path[-1].append('bombe')
                    move = move_path(myid, path)
                else:
                    move = put_bomb(myid)
                print('[Enemy p' + str(res.get('id')) + ' stopped] move', move)
                await websocket.send(move)

async def hello(uri, player_name, nb_players):
    try:
        async with websockets.connect(uri) as websocket:
            await play(websocket, player_name, nb_players)
    except ConnectionClosedError:
        print('[ConnectionClosedError] Quit program')
        quit()

def main(player_name, nb_players=1, game_name="dorogame"):
    asyncio.get_event_loop().run_until_complete(
        hello('wss://pi-univers.fr/websocketBomber/?t=play&s='+ game_name +'&nbj='+str(nb_players)+'&name=' + player_name, player_name, nb_players))

if __name__ == "__main__":
    len_args = len(argv)
    if len_args < 2 or len_args > 4:
        print('[ArgumentError] The arguments of the program should be: player_name, nb_players (optional, default 1), game_name (optional, default dorogame)')
    else:
        nb_players = 0
        try:
            if len_args == 2:
                main(argv[1])
            else:
                nb_players = int(argv[2])
                if nb_players <= 0:
                    raise ValueError
                elif len_args == 3:
                    main(argv[1], nb_players)
                else:
                    main(argv[1], nb_players, argv[3])
        except ValueError:
            print('[ArgumentError] The number of players (2nd argument) should be a positive non-zero integer.')
