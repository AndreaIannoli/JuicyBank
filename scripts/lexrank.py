MAX_CHAR = 256

# Factorial


def fact(n):
    res = 1
    for i in range(2, n + 1):
        res *= i
    return res

# Construct a count array where value at every index
# contains count of smaller characters in whole string


def populate_and_increase_count(count, s):
    for ch in s:
        count[ord(ch)] += 1

    for i in range(1, MAX_CHAR):
        count[i] += count[i - 1]

# Removes a character ch from count[] array
# constructed by populate_and_increase_count()


def update_count(count, ch):
    for i in range(ord(ch), MAX_CHAR):
        count[i] -= 1

# A function to find rank of a string in all permutations
# of characters


def find_rank(s):
    n = len(s)
    mul = fact(n)
    rank = 1

    # All elements of count[] are initialized with 0
    count = [0] * MAX_CHAR

    # Populate the count array such that count[i]
    # contains count of characters which are present
    # in s and are smaller than i
    populate_and_increase_count(count, s)

    for i in range(n):
        mul //= (n - i)

        # Count number of chars smaller than s[i]
        # from s[i+1] to s[len-1]
        rank += count[ord(s[i]) - 1] * mul

        # Reduce count of characters greater than s[i]
        update_count(count, s[i])

    return rank
