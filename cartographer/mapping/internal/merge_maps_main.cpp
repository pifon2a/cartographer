/*
 * Copyright 2018 The Cartographer Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cmath>
#include <fstream>
#include <string>

#include "cartographer/common/port.h"
#include "cartographer/io/file_writer.h"
#include "cartographer/io/image.h"
#include "cartographer/io/proto_stream.h"
#include "cartographer/io/submap_painter.h"
#include "cartographer/mapping/internal/2d/overlapping_submaps_trimmer_2d.h"
#include "cartographer/mapping/internal/2d/pose_graph_2d.h"
#include "cartographer/mapping/internal/3d/pose_graph_3d.h"
#include "cartographer/mapping/internal/testing/test_helpers.h"
#include "cartographer/mapping/map_builder.h"
#include "cartographer/mapping/pose_graph.h"
#include "cartographer/mapping/proto/pose_graph.pb.h"
#include "cartographer/mapping/proto/trajectory_builder_options.pb.h"
#include "cartographer/transform/transform.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_bool(use_3d, false, "Use 3D pipeline (default is 2D).");
DEFINE_string(pose_graph_filenames, "",
              "Comma-separated list of pbstream files to read.");
DEFINE_bool(skip_optimization, false,
            "Skip all constraint building and optimization.");
DEFINE_double(resolution, 0.05, "Size of a pixel (meters) in output image.");

DEFINE_bool(use_trimmer, false, "ff");
DEFINE_double(fresh_submaps_count, 1, "fasdf");
DEFINE_double(min_covered_area, 2, "sd");

namespace cartographer {
namespace mapping {
namespace {

std::vector<std::string> SplitString(const std::string& input,
                                     const char delimiter) {
  std::istringstream stream(input);
  std::string token;
  std::vector<std::string> tokens;
  while (std::getline(stream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

class EmptyOptimizationProblem2D : public pose_graph::OptimizationProblem2D {
 public:
  EmptyOptimizationProblem2D()
      : OptimizationProblem2D(pose_graph::proto::OptimizationProblemOptions{}) {
  }
  ~EmptyOptimizationProblem2D() override = default;
  void Solve(
      const std::vector<Constraint>& constraints,
      const std::set<int>& frozen_trajectories,
      const std::map<std::string, LandmarkNode>& landmark_nodes) override {}
};

class EmptyOptimizationProblem3D : public pose_graph::OptimizationProblem3D {
 public:
  EmptyOptimizationProblem3D()
      : OptimizationProblem3D(pose_graph::proto::OptimizationProblemOptions{}) {
  }
  ~EmptyOptimizationProblem3D() override = default;
  void Solve(
      const std::vector<Constraint>& constraints,
      const std::set<int>& frozen_trajectories,
      const std::map<std::string, LandmarkNode>& landmark_nodes) override {}
};

class RemappingTrajectoryImporter {
 public:
  explicit RemappingTrajectoryImporter(mapping::PoseGraph* pose_graph)
      : pose_graph_(pose_graph) {}

  void LoadTrajectories(io::ProtoStreamReaderInterface* reader);

  void SerializeState(io::ProtoStreamWriter* writer);

 private:
  std::vector<proto::TrajectoryBuilderOptionsWithSensorIds>
      all_trajectory_builder_options_;
  int next_trajectory_id_ = 0;
  mapping::PoseGraph* pose_graph_;
};

void RemappingTrajectoryImporter::LoadTrajectories(
    io::ProtoStreamReaderInterface* const reader) {
  // TODO(gaschler): Reuse this function with MapBuilder.
  proto::PoseGraph pose_graph_proto;
  CHECK(reader->ReadProto(&pose_graph_proto));
  proto::AllTrajectoryBuilderOptions all_builder_options_proto;
  CHECK(reader->ReadProto(&all_builder_options_proto));
  bool is_pose_graph_3d = dynamic_cast<PoseGraph3D*>(pose_graph_) != nullptr;
  CHECK(all_builder_options_proto.options_with_sensor_ids_size() > 0);
  auto& o = all_builder_options_proto.options_with_sensor_ids(0);
  CHECK(o.has_trajectory_builder_options());
  if (is_pose_graph_3d) {
    CHECK(o.trajectory_builder_options().has_trajectory_builder_3d_options());
  } else {
    CHECK(o.trajectory_builder_options().has_trajectory_builder_2d_options());
  }
  CHECK_EQ(pose_graph_proto.trajectory_size(),
           all_builder_options_proto.options_with_sensor_ids_size());

  std::map<int, int> trajectory_remapping;
  for (auto& trajectory_proto : *pose_graph_proto.mutable_trajectory()) {
    const auto& options_with_sensor_ids_proto =
        all_builder_options_proto.options_with_sensor_ids(
            trajectory_proto.trajectory_id());
    all_trajectory_builder_options_.push_back(options_with_sensor_ids_proto);
    CHECK(trajectory_remapping
              .emplace(trajectory_proto.trajectory_id(), next_trajectory_id_)
              .second)
        << "Duplicate trajectory ID: " << trajectory_proto.trajectory_id();
    trajectory_proto.set_trajectory_id(next_trajectory_id_);
    next_trajectory_id_++;
  }

  // Apply the calculated remapping to constraints in the pose graph proto.
  for (auto& constraint_proto : *pose_graph_proto.mutable_constraint()) {
    constraint_proto.mutable_submap_id()->set_trajectory_id(
        trajectory_remapping.at(constraint_proto.submap_id().trajectory_id()));
    constraint_proto.mutable_node_id()->set_trajectory_id(
        trajectory_remapping.at(constraint_proto.node_id().trajectory_id()));
  }

  MapById<SubmapId, transform::Rigid3d> submap_poses;
  for (const proto::Trajectory& trajectory_proto :
       pose_graph_proto.trajectory()) {
    for (const proto::Trajectory::Submap& submap_proto :
         trajectory_proto.submap()) {
      submap_poses.Insert(SubmapId{trajectory_proto.trajectory_id(),
                                   submap_proto.submap_index()},
                          transform::ToRigid3(submap_proto.pose()));
    }
  }

  MapById<NodeId, transform::Rigid3d> node_poses;
  for (const proto::Trajectory& trajectory_proto :
       pose_graph_proto.trajectory()) {
    for (const proto::Trajectory::Node& node_proto : trajectory_proto.node()) {
      node_poses.Insert(
          NodeId{trajectory_proto.trajectory_id(), node_proto.node_index()},
          transform::ToRigid3(node_proto.pose()));
    }
  }

  for (;;) {
    proto::SerializedData proto;
    if (!reader->ReadProto(&proto)) {
      break;
    }

    if (proto.has_node()) {
      proto.mutable_node()->mutable_node_id()->set_trajectory_id(
          trajectory_remapping.at(proto.node().node_id().trajectory_id()));
      const transform::Rigid3d node_pose =
          node_poses.at(NodeId{proto.node().node_id().trajectory_id(),
                               proto.node().node_id().node_index()});
      pose_graph_->AddNodeFromProto(node_pose, proto.node());
    }
    if (proto.has_submap()) {
      if (is_pose_graph_3d) {
        CHECK(proto.submap().has_submap_3d());
      } else {
        CHECK(proto.submap().has_submap_2d());
      }
      proto.mutable_submap()->mutable_submap_id()->set_trajectory_id(
          trajectory_remapping.at(proto.submap().submap_id().trajectory_id()));
      const transform::Rigid3d submap_pose =
          submap_poses.at(SubmapId{proto.submap().submap_id().trajectory_id(),
                                   proto.submap().submap_id().submap_index()});
      pose_graph_->AddSubmapFromProto(submap_pose, proto.submap());
    }
    if (proto.has_trajectory_data()) {
      proto.mutable_trajectory_data()->set_trajectory_id(
          trajectory_remapping.at(proto.trajectory_data().trajectory_id()));
      pose_graph_->SetTrajectoryDataFromProto(proto.trajectory_data());
    }
    if (proto.has_imu_data()) {
      pose_graph_->AddImuData(
          trajectory_remapping.at(proto.imu_data().trajectory_id()),
          sensor::FromProto(proto.imu_data().imu_data()));
    }
    if (proto.has_odometry_data()) {
      pose_graph_->AddOdometryData(
          trajectory_remapping.at(proto.odometry_data().trajectory_id()),
          sensor::FromProto(proto.odometry_data().odometry_data()));
    }
    if (proto.has_fixed_frame_pose_data()) {
      pose_graph_->AddFixedFramePoseData(
          trajectory_remapping.at(
              proto.fixed_frame_pose_data().trajectory_id()),
          sensor::FromProto(
              proto.fixed_frame_pose_data().fixed_frame_pose_data()));
    }
    if (proto.has_landmark_data()) {
      pose_graph_->AddLandmarkData(
          trajectory_remapping.at(proto.landmark_data().trajectory_id()),
          sensor::FromProto(proto.landmark_data().landmark_data()));
    }
  }

  pose_graph_->AddSerializedConstraints(
      FromProto(pose_graph_proto.constraint()));
  if (FLAGS_use_trimmer) {
    pose_graph_->AddTrimmer(common::make_unique<OverlappingSubmapsTrimmer2D>(
        FLAGS_fresh_submaps_count,
        FLAGS_min_covered_area / common::Pow2(FLAGS_resolution), 1));
  }
  CHECK(reader->eof());
}

void RemappingTrajectoryImporter::SerializeState(
    io::ProtoStreamWriter* const writer) {
  MapBuilder::SerializeState(all_trajectory_builder_options_, pose_graph_,
                             writer);
}

void WritePng(PoseGraphInterface* pose_graph, io::StreamFileWriter* png_writer,
              double image_resolution) {
  LOG(INFO) << "Loading submap slices from serialized data.";
  std::map<SubmapId, io::SubmapSlice> submap_slices;
  auto all_submap_data = pose_graph->GetAllSubmapData();
  for (const auto& submap_data : all_submap_data) {
    proto::Submap submap_proto;
    submap_data.data.submap->ToProto(&submap_proto, true);
    FillSubmapSlice(submap_data.data.pose, submap_proto,
                    &submap_slices[submap_data.id]);
  }
  LOG(INFO) << "Generating combined map image from submap slices";
  auto result = io::PaintSubmapSlices(submap_slices, image_resolution);
  io::Image image(std::move(result.surface));
  image.WritePng(png_writer);
  LOG(INFO) << "Wrote image to " << png_writer->GetFilename();
}

void PrintReport(PoseGraphInterface* pose_graph) {
  LOG(INFO) << "Summary of inter-trajectory constraints:";
  const auto trajectory_node_poses = pose_graph->GetTrajectoryNodePoses();
  const auto submap_poses = pose_graph->GetAllSubmapPoses();
  const auto constraints = pose_graph->constraints();
  cartographer::common::Histogram residual_inter_translation,
      residual_inter_rotation;
  const float outlier_residual_translation = 0.1f;
  std::map<cartographer::mapping::SubmapId, int> submap_to_num_constraints;
  std::map<cartographer::mapping::SubmapId, int> submap_to_num_good_constraints;
  for (const auto& submap : submap_poses) {
    submap_to_num_constraints[submap.id] = 0;
    submap_to_num_good_constraints[submap.id] = 0;
  }
  for (const auto& constraint : constraints) {
    if (constraint.tag == cartographer::mapping::PoseGraphInterface::
                              Constraint::Tag::INTRA_SUBMAP ||
        constraint.node_id.trajectory_id ==
            constraint.submap_id.trajectory_id) {
      continue;
    }
    const auto submap_it = submap_poses.find(constraint.submap_id);
    if (submap_it == submap_poses.end()) {
      continue;
    }
    const auto& submap_pose = submap_it->data.pose;
    const auto node_it = trajectory_node_poses.find(constraint.node_id);
    if (node_it == trajectory_node_poses.end()) {
      continue;
    }
    const transform::Rigid3d& trajectory_node_pose = node_it->data.global_pose;
    const transform::Rigid3d constraint_pose =
        submap_pose * constraint.pose.zbar_ij;
    float residual_translation =
        (trajectory_node_pose.translation() - constraint_pose.translation())
            .norm();
    float residual_rotation = trajectory_node_pose.rotation().angularDistance(
        constraint_pose.rotation());
    residual_inter_translation.Add(residual_translation);
    residual_inter_rotation.Add(residual_rotation);
    submap_to_num_constraints.at(submap_it->id)++;
    if (residual_translation < outlier_residual_translation) {
      submap_to_num_good_constraints.at(submap_it->id)++;
    }
  }
  LOG(INFO) << "translation residuals: "
            << residual_inter_translation.ToString(10);
  LOG(INFO) << "rotation residuals: " << residual_inter_rotation.ToString(10);
  cartographer::common::Histogram constraints_per_submap,
      good_constraints_per_submap, good_minus_bad_constraints_per_submap;
  for (const auto& pair : submap_to_num_constraints) {
    constraints_per_submap.Add(pair.second);
  }
  for (const auto& pair : submap_to_num_good_constraints) {
    good_constraints_per_submap.Add(pair.second);
  }
  for (const auto& submap : submap_poses) {
    int good_minus_bad = 2 * submap_to_num_good_constraints.at(submap.id) -
                         submap_to_num_constraints.at(submap.id);
    good_minus_bad_constraints_per_submap.Add(good_minus_bad);
  }
  LOG(INFO) << "#constraints per submap: "
            << constraints_per_submap.ToString(10);
  LOG(INFO) << "#good constraints per submap: "
            << good_constraints_per_submap.ToString(10);
  LOG(INFO) << "#good - #bad constraints per submap: "
            << good_minus_bad_constraints_per_submap.ToString(10);
  LOG(INFO) << "Here, \"good\" means the residual translation is lower than "
            << outlier_residual_translation;
}

void Run(bool use_3d, const std::string& pose_graph_filenames,
         bool skip_optimization, double image_resolution) {
  auto filenames = SplitString(pose_graph_filenames, ',');

  // TODO: Read options from flag rather than using default.
  auto thread_pool = common::make_unique<common::ThreadPool>(16);
  const std::string kPoseGraphLua = R"text(
      include "pose_graph.lua"
      return POSE_GRAPH)text";
  auto pose_graph_parameters = test::ResolveLuaParameters(kPoseGraphLua);
  auto pose_graph_options = CreatePoseGraphOptions(pose_graph_parameters.get());
  pose_graph_options.mutable_optimization_problem_options()
      ->set_log_solver_summary(true);

  std::unique_ptr<mapping::PoseGraph> pose_graph;
  if (use_3d) {
    std::unique_ptr<pose_graph::OptimizationProblem3D> optimization_problem =
        (skip_optimization)
            ? common::make_unique<EmptyOptimizationProblem3D>()
            : common::make_unique<pose_graph::OptimizationProblem3D>(
                  pose_graph_options.optimization_problem_options());
    pose_graph = common::make_unique<PoseGraph3D>(
        pose_graph_options, std::move(optimization_problem), thread_pool.get());
  } else {
    std::unique_ptr<pose_graph::OptimizationProblem2D> optimization_problem =
        (skip_optimization)
            ? common::make_unique<EmptyOptimizationProblem2D>()
            : common::make_unique<pose_graph::OptimizationProblem2D>(
                  pose_graph_options.optimization_problem_options());
    pose_graph = common::make_unique<PoseGraph2D>(
        pose_graph_options, std::move(optimization_problem), thread_pool.get());
  }

  RemappingTrajectoryImporter remapping_importer(pose_graph.get());
  for (const std::string& pose_graph_filename : filenames) {
    LOG(INFO) << "Reading pose graph from '" << pose_graph_filename << "'...";
    io::ProtoStreamReader reader(pose_graph_filename);
    remapping_importer.LoadTrajectories(&reader);
  }
  pose_graph->RunFinalOptimization();
  CHECK(!pose_graph->IsTrajectoryFrozen(0));

  if (!skip_optimization) {
    pose_graph->FindInterTrajectoryGlobalConstraints();
    pose_graph->RunFinalOptimization();
    for (int i = 0; i < 10; ++i) {
      for (int j = 0; j < 20; ++j) {
        pose_graph->FindInterTrajectoryConstraints();
      }
      pose_graph->RunFinalOptimization();
    }
    PrintReport(pose_graph.get());
  }
  {
    std::string output_filename = "merged.pbstream";
    io::ProtoStreamWriter writer(output_filename);
    remapping_importer.SerializeState(&writer);
    LOG(INFO) << "Wrote merged pbstream " << output_filename;
  }
  {
    io::StreamFileWriter writer("merged.png");
    WritePng(pose_graph.get(), &writer, image_resolution);
  }
}

}  // namespace
}  // namespace mapping
}  // namespace cartographer

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;
  google::SetUsageMessage(
      "\n\n"
      "This program merge multiple pose graph pbstream files.\n");
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_pose_graph_filenames.empty()) {
    google::ShowUsageWithFlags(argv[0]);
    return EXIT_FAILURE;
  }
  ::cartographer::mapping::Run(FLAGS_use_3d, FLAGS_pose_graph_filenames,
                               FLAGS_skip_optimization, FLAGS_resolution);
}
