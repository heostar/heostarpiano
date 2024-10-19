import atexit
import pyperclip

def join_with_space(*args):
    """
    Joins multiple strings with a space in between.
    
    Args:
        *args: Any number of string arguments.
        
    Returns:
        str: The joined string.
    """
    return ' '.join(args)

def main():
    
    tags = []

    ge = input("Type game name in english: ")
    se = input("Type song name in english: ")    
    ae = input("Type artist name in english: ")
    gk = input("Type game name in korean: ")
    sk = input("Type song name in korean: ")    
    ak = input("Type artist name in korean: ")
    pe = "piano"
    ce = "cover"
    pk = "피아노"
    ck = "커버"

    gamename_yes = int(input("Is there another game name for the song ? Type 1 if yes, 0 for no: "))
    gge = None
    ggk = None
    if gamename_yes == 1:
        gge = input("Type another game name in english: ")
        ggk = input("Type another game name in Korean: ")

    nick_yes = int(input("Is there a nickname for the song ? Type 1 if yes, 0 for no: "))
    nn = None
    if nick_yes == 1:
        nn = input("Type nickname in korean: ")
    
    tags.append(join_with_space(ge, se, pe, ce))
    tags.append(join_with_space(ge, se, pe))
    tags.append(join_with_space(ge, se))
    tags.append(ge)
    
    tags.append(join_with_space(se, pe, ce))
    tags.append(join_with_space(se, pe))
    tags.append(se)

    tags.append(join_with_space(ae, pe, ce))
    tags.append(join_with_space(ae, pe))
    tags.append(ae)

    tags.append(ak)

    tags.append(join_with_space(gk, sk, pk, ck))
    tags.append(join_with_space(gk, sk, pk))
    tags.append(join_with_space(gk, sk))
    tags.append(gk)

    if gamename_yes == 1:
        tags.append(join_with_space(gge, se, pe, ce))
        tags.append(join_with_space(gge, se, pe))
        tags.append(join_with_space(gge, se))
        tags.append(gge)

        tags.append(join_with_space(ggk, sk, pk, ck))
        tags.append(join_with_space(ggk, sk, pk))
        tags.append(join_with_space(ggk, sk))
        tags.append(ggk)
    
    if nick_yes == 1:
        tags.append(join_with_space(nn, pk, ck))
        tags.append(join_with_space(gk, nn, pk))
        tags.append(nn)

    pyperclip.copy(','.join(tags))

    print("tags copied to the clipboard")
    

if __name__ == "__main__":
    main()