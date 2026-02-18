from fl_app.task import prepare_federated_dataset

fed = prepare_federated_dataset(
    dataset="mnist",        #имя датасета
    num_clients=10,         #число клиентов
    scheme="iid",           #схема разбиения (iid или dirichlet)
    #alpha=0.3,              #параметр alpha для схемы dirichlet (если scheme="iid", закоментируй) 
    save_plot_to="reports/mnist_iid.png",  #куда сохранять график распределения классов по клиентам
)

client0_ds = fed.get_partition(0)
print(client0_ds[0].keys())      # например: img/label
print("label column:", fed.label_name)
print("plot:", fed.plot_path)