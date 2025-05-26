# modelling_lib

The point of this library is to contain the model objects and functions used *on* on the data.
Will contain anything related to instantiating & fitting models, as well as generating model predictions.

TODO:

- [x] Instead of replacing shared leaves with `0`, replace with some class/object instead
- [ ] Nicer `__repr__` for `ShareModule` that actually says the memory address
- [ ] Add memory address to the top of `print_model_tree`
- [ ] Support tuples, lists and dicts of models as attributes of models
- [ ] Handle non-odd number of modes
- [ ] Write better tests
- [ ] Rigorously type check the tests
