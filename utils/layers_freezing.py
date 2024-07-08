def freeze_layers(model, layer, test_input):

    # Define a hook function
    freeze_flag = True
    parameters_to_be_frozen = []
    def hook(module, input, output):
        nonlocal freeze_flag
        if module is layer:
            for name, parameters in module.named_parameters():
                parameters_to_be_frozen.append(parameters)
            freeze_flag = False

        if freeze_flag:
            for name, parameters in module.named_parameters():
                if name.find('.') == -1:
                    parameters_to_be_frozen.append(parameters)

    # Register a forward hook for all the modules
    hooks_handles = {}
    for name, module in model.named_modules():
        hooks_handles[name] = module.register_forward_hook(hook)
    
    if model.training:
        model.eval()
        model(test_input)
        model.train()
    else:
        model(test_input)

    # Freeze the selected parameters
    for parameters in parameters_to_be_frozen:
        parameters.requires_grad = False

    # Remove the registered forward hooks for all the modules
    for _, handle in hooks_handles.items():
        handle.remove()