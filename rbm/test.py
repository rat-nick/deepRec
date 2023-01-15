from . import model

if __name__ == "__main__":
    rbm = model.Model((2, 3), (5,))
    rbm.summarize()
    print(rbm.latestMAE)
    print(rbm.latestRSME)
    print(rbm.current_epoch)
    rbm.next_epoch()
    rbm.next_epoch()
    rbm.next_epoch()
    print(rbm.current_epoch)
    print(rbm.latestMAE)
    print(rbm.latestRSME)
    print(rbm.current_epoch)
    for param in rbm.named_parameters():
        print(param)
