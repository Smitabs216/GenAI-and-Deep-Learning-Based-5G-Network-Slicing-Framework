def select_slice(tp, delay, plr):

    # URLLC
    if delay < 10 and plr < 0.0001:
        return "URLLC"

    # eMBB
    elif tp > 10 and delay < 200:
        return "eMBB"

    # mMTC
    else:
        return "mMTC"