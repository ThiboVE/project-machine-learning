
    # Save mean validation performance
    performance_dict[tuple(params.items())] = np.mean(inner_fold_mae)

# Selecting the best inner CV params
best_params = min(performance_dict, key=performance_dict.get)
best_params = dict(best_params)


    # ================================
    # RETRAIN ON FULL TRAIN+VALIDATION
    # ================================
    full_train_loader = DataLoader(
        dataset,
        batch_size=best_params["batch_size"],
        sampler=SubsetRandomSampler(train_val_idx),
        collate_fn=collate_graph_dataset,
    )
    
    test_loader = DataLoader(
        dataset,
        batch_size=best_params["batch_size"],
        sampler=SubsetRandomSampler(test_idx),
        collate_fn=collate_graph_dataset
    )

    model = ChemGCN(
        node_vec_len=node_vec_len,
        node_fea_len=best_params["hidden_nodes"],
        hidden_fea_len=best_params["hidden_nodes"],
        n_conv=best_params["n_conv_layers"],
        n_hidden=best_params["n_hidden_layers"],
        n_outputs=1,
        p_dropout=0.1,
    )
    if use_GPU:
        model.cuda()

    outputs = [dataset[i][1] for i in train_val_idx]
    standardizer = Standardizer(torch.Tensor(outputs))

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["learning_rate"])
    loss_fn = torch.nn.L1Loss()

    for epoch in range(n_epochs):
        train_model(
            epoch, model, full_train_loader, optimizer, loss_fn,
            standardizer, use_GPU, max_atoms, node_vec_len
        )

    # ================================
    # FINAL TEST ON OUTER FOLD
    # ================================
    test_loss, test_mae = test_model(
        model, test_loader, loss_fn, standardizer,
        use_GPU, max_atoms, node_vec_len
    )

    outer_results.append(test_mae)
    print(f"===== Outer Fold {outer_fold+1} Test MAE: {test_mae:.4f} =====")


