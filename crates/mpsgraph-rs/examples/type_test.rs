use mpsgraph::{
    core::MPSDataType,
    data_types::{MPSGraphShapedType, MPSGraphType},
    shape::MPSShape,
};

fn main() {
    // Create a basic type
    let graph_type = MPSGraphType::new();
    println!("Basic graph type description: {}", graph_type.description());

    // Create a shape for our shaped type
    let shape = MPSShape::from_slice(&[2, 3, 4]);
    println!("Shape: {:?}", shape);

    // Create a shaped type with the shape and data type
    let shaped_type = MPSGraphShapedType::new(&shape, MPSDataType::Float32);
    println!("Shaped type: {:?}", shaped_type);
    println!("  Data type: {:?}", shaped_type.data_type());
    println!("  Rank: {}", shaped_type.rank());
    println!("  Is ranked: {}", shaped_type.is_ranked());

    // Create a tensor type with rank
    let ranked_type = MPSGraphShapedType::tensor_type_with_rank(3, MPSDataType::Float16);
    println!("Ranked tensor type: {:?}", ranked_type);

    // Create an unranked tensor type
    let unranked_type = MPSGraphShapedType::unranked_tensor_type(MPSDataType::Int32);
    println!("Unranked tensor type: {:?}", unranked_type);

    println!("Type system test completed successfully!");
}
