A = 7
B = 4
GOAL = 5
visited = set()
print("Enter rule numbers in the order you want to apply (space separated)")
print("1.Fill X  2.Fill Y  3.Empty X  4.Empty Y")
print("5.X→Y (X empty)  6.X→Y (Y full)")
print("7.Y→X (Y empty)  8.Y→X (X full)")
rule_order = list(map(int, input("Enter rules: ").split()))
def apply_rule(rule, state):
    x, y = state
    if rule == 1:
        return "Rule 1: Fill X", (A, y)

    elif rule == 2:
        return "Rule 2: Fill Y", (x, B)

    elif rule == 3:
        return "Rule 3: Empty X", (0, y)

    elif rule == 4:
        return "Rule 4: Empty Y", (x, 0)

    elif rule == 5 and x > 0 and x <= (B - y):
        return "Rule 5: Pour X → Y (X empty)", (0, y + x)

    elif rule == 6 and x > (B - y):
        return "Rule 6: Pour X → Y (Y full)", (x - (B - y), B)

    elif rule == 7 and y > 0 and y <= (A - x):
        return "Rule 7: Pour Y → X (Y empty)", (x + y, 0)

    elif rule == 8 and y > (A - x):
        return "Rule 8: Pour Y → X (X full)", (A, y - (A - x))

    return None, None


def dfs(state, path):
    x, y = state

    if x == GOAL:
        print("\nSolution Found!\n")
        for step in path:
            print(step)
        print(f"\nFinal State: {state}")
        return True

    if state in visited:
        return False

    visited.add(state)

    for rule in rule_order:
        rule_name, new_state = apply_rule(rule, state)
        if new_state and new_state not in visited:
            if dfs(new_state, path + [f"{rule_name}: {state} → {new_state}"]):
                return True

    return False
initial_state = (0, 0)
dfs(initial_state, [])
